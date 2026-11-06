// llama-iolbatch — JSONL continuous-batch driver for parallel vote decoding.
//
// Adapted from llama.cpp examples/parallel/parallel.cpp @ b9964 (client/slot
// structure, chunked batch views, llama_memory_seq_cp prefix sharing) — see
// that file for the upstream provenance. Written for the IOL-AI 2026 solver:
// the eval sandbox seccomp kills every socket syscall (TCP loopback, AF_UNIX,
// socketpair all EPERM) but stdin/stdout pipes survive, so this speaks JSONL
// over pipes instead of HTTP.
//
// Protocol (one JSON object per line):
//   stdin : {"reqs": [{"id": str, "prompt": str, "n_samples": int,
//                      "max_new": int, "temp": float, "top_p": float,
//                      "seed_base": int, "logprobs": bool}, ...]}
//           {"cmd": "quit"}            -> clean exit (EOF works too)
//   stdout: {"ready": true, "n_ctx": int, "slots": int}       after model load
//           {"id": str, "sample_idx": int, "text": str,
//            "n_prompt": int, "n_decoded": int, "stop": str}  per sample
//           {"done": true, "n_results": int, "t_ms": int}     per group
//
// With "logprobs": true a sample also carries "lps" (raw-model logprob per
// emitted token), "tlens" (that token's piece length in BYTES), "lp_sum" and
// "lp_eos". lps/tlens are parallel arrays covering exactly the bytes of "text"
// (the EOG token is scored into lp_eos but never appended), so the client can
// map any substring of the answer back to its tokens by cumulative byte offset
// and weight votes by per-ITEM confidence (E33).
//
// All n_samples of one request share a single prompt prefill: the prompt is
// decoded once on the request's first slot, then llama_memory_seq_cp clones
// the prefix KV to the sibling slots (zero-copy: the KV cache is forced
// unified). Requests are admitted whole (all samples together) while free
// slots and context-cell budget allow; decoding is continuous across requests.
//
// stdout hygiene: common's LOG_INF writes to fd 1, so we dup the real stdout
// for the protocol and point fd 1 at stderr before initializing anything.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct request {
    std::string id;
    std::string prompt;
    int n_samples = 1;
    int max_new   = 128;
    float temp    = 0.8f;
    float top_p   = 0.95f;
    int seed_base = 42;
    bool logprobs = false;  // emit per-token raw-model logprobs (E33 vote weighting)

    std::vector<llama_token> tokens;  // filled at admission time
    int n_active = 0;                 // streams currently decoding
    int n_pending = 0;                // samples not yet admitted (always admitted whole)
    // set at admission, consumed by the fork step inside the decode loop:
    int i_last = -1;                  // batch index of the prompt's last token
    llama_seq_id seq_prompt = -1;     // sequence that hosts the shared prompt prefill
    bool forked = false;              // prefix KV copied to the sibling sequences
};

struct stream {
    int  req_idx    = -1;   // index into the group's request vector; -1 = free slot
    int  sample_idx = 0;
    llama_seq_id seq_id = -1;  // the slot's fixed sequence id
    llama_token sampled = 0;   // last sampled token (fed back on the next step)
    int  n_past     = 0;
    int  n_prompt   = 0;
    int  n_decoded  = 0;    // tokens generated so far (sampled, incl. a final EOG)
    int  i_batch    = -1;
    int  max_new    = 0;
    std::string response;
    common_sampler * smpl = nullptr;
    // E33 logprobs: parallel arrays over the tokens APPENDED to response
    std::vector<float> lps;    // raw-model logprob of each emitted token
    std::vector<int>   tlens;  // that token's piece length in bytes
    double lp_sum   = 0.0;     // sum over emitted tokens (excludes the EOG)
    float  lp_eos   = 0.0f;    // logprob of the stop token, when it stopped on one
};

static FILE * g_proto = nullptr;  // the real stdout (protocol channel)

static void proto_emit(const json & j) {
    // error_handler_t::replace: generation can stop mid-multibyte sequence
    const std::string s = j.dump(-1, ' ', false, json::error_handler_t::replace);
    fprintf(g_proto, "%s\n", s.c_str());
    fflush(g_proto);
}

// every line of a group's output carries the client-chosen group tag, so the
// client can discard stale frames after a timeout/desync (results of an
// abandoned group must never be attributed to a later one)
static int64_t g_group = 0;

// Logprob of the sampled token under the RAW model distribution — deliberately
// NOT the sampler's post-temperature/top-p probability: the sampler's number
// says "how likely was this draw given the knobs", the raw one says "how much
// does the model believe this token", which is the quantity vote weighting
// wants (and it stays comparable across temp settings). One online-rescaled
// logsumexp pass over the vocab; ~600 KB of already-resident logits per token.
static float token_logprob(const float * logits, int n_vocab, llama_token tok) {
    if (!logits || n_vocab <= 0 || tok < 0 || tok >= n_vocab) {
        return 0.0f;
    }
    float  max_l = logits[0];
    double sum   = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        const float l = logits[i];
        if (l > max_l) {
            sum *= std::exp((double) max_l - (double) l);  // rescale to the new max
            max_l = l;
        }
        sum += std::exp((double) l - (double) max_l);
    }
    return (float) ((double) logits[tok] - ((double) max_l + std::log(sum)));
}

static void emit_result(const request & req, const stream & st, const char * stop) {
    json j{
        {"group", g_group},
        {"id", req.id},
        {"sample_idx", st.sample_idx},
        {"text", st.response},
        {"n_prompt", st.n_prompt},
        {"n_decoded", st.n_decoded},
        {"stop", stop},
    };
    if (req.logprobs) {
        // 4 decimals: ~1e-4 nats is far below any weighting effect and keeps the
        // per-group payload at tens of KB instead of hundreds
        json lps = json::array();
        for (const float lp : st.lps) {
            lps.push_back(std::round(lp * 10000.0f) / 10000.0f);
        }
        j["lps"]    = lps;
        j["tlens"]  = st.tlens;
        j["lp_sum"] = std::round(st.lp_sum * 10000.0) / 10000.0;
        j["lp_eos"] = std::round(st.lp_eos * 10000.0f) / 10000.0f;
    }
    proto_emit(j);
}

static void emit_error(const request & req, int sample_idx, const std::string & msg) {
    proto_emit(json{
        {"group", g_group},
        {"id", req.id},
        {"sample_idx", sample_idx},
        {"text", ""},
        {"n_prompt", 0},
        {"n_decoded", 0},
        {"stop", "error"},
        {"error", msg},
    });
}

int main(int argc, char ** argv) {
    // claim the protocol channel before anything can print to fd 1
    g_proto = fdopen(dup(STDOUT_FILENO), "w");
    if (!g_proto) {
        fprintf(stderr, "iolbatch: cannot dup stdout\n");
        return 1;
    }
    dup2(STDERR_FILENO, STDOUT_FILENO);  // library logs (LOG_INF hits fd 1) -> stderr

    common_params params;
    // (no params.n_predict: per-stream length comes from each request's max_new)

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PARALLEL)) {
        return 1;
    }

    // shared-prefix seq_cp must be zero-copy -> unified KV pool across slots
    params.kv_unified = true;

    const int n_slots = std::max(1, params.n_parallel);  // == n_seq_max (common.cpp:1594)

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "iolbatch: model/context init failed\n");
        return 1;
    }

    llama_memory_t      mem   = llama_get_memory(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int n_ctx    = (int) llama_n_ctx(ctx);
    // slack cells so admission never lands exactly on the rim — must be >= n_slots
    // so one iteration's batch (admitted prompts + one continuation token per slot)
    // can never exceed the llama_batch capacity of n_ctx tokens
    const int ctx_margin = 16 + n_slots;

    // batch capacity = n_ctx tokens: admission control keeps total resident tokens
    // under n_ctx, so one group iteration can never queue more than that
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    std::vector<stream> slots(n_slots);
    for (int i = 0; i < n_slots; ++i) {
        slots[i].seq_id = i;
    }

    proto_emit(json{{"ready", true}, {"n_ctx", n_ctx}, {"slots", n_slots}});

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        json in = json::parse(line, nullptr, /*allow_exceptions=*/false);
        if (in.is_discarded() || !in.is_object()) {
            proto_emit(json{{"group", -1}, {"done", true}, {"n_results", 0}, {"error", "bad request json"}});
            continue;
        }
        if (in.value("cmd", "") == std::string("quit")) {
            break;
        }
        g_group = in.value("group", (int64_t) 0);

        // ---- parse the group ----
        std::vector<request> reqs;
        for (const auto & r : in.value("reqs", json::array())) {
            request q;
            q.id        = r.value("id", "");
            q.prompt    = r.value("prompt", "");
            q.n_samples = std::max(1, r.value("n_samples", 1));
            q.max_new   = std::max(1, r.value("max_new", 128));
            q.temp      = r.value("temp", 0.8f);
            q.top_p     = r.value("top_p", 0.95f);
            q.seed_base = r.value("seed_base", 42);
            q.logprobs  = r.value("logprobs", false);
            q.n_pending = q.n_samples;
            reqs.push_back(std::move(q));
        }

        const auto t_group_start = ggml_time_us();
        int n_results = 0;
        size_t next_req = 0;   // requests are admitted in order
        int n_ctx_used  = 0;   // reserved cells: prompts of live requests + max_new per stream
        int n_free_slots = n_slots;
        bool fatal = false;

        // reject requests that can never fit, up front
        for (auto & q : reqs) {
            q.tokens = common_tokenize(ctx, q.prompt, /*add_special=*/true, /*parse_special=*/true);
            const int need = (int) q.tokens.size() + q.n_samples * q.max_new;
            // an empty prompt would make i_last point at another request's token
            if (q.tokens.empty() || q.n_samples > n_slots || need > n_ctx - ctx_margin) {
                for (int s = 0; s < q.n_samples; ++s) {
                    emit_error(q, s, "request cannot fit: " + std::to_string(q.tokens.size()) +
                                     " prompt + " + std::to_string(q.n_samples) + "x" +
                                     std::to_string(q.max_new) + " gen vs n_ctx " +
                                     std::to_string(n_ctx) + ", slots " + std::to_string(n_slots));
                    n_results++;
                }
                q.n_pending = 0;  // skip at admission
            }
        }

        while (!fatal) {
            common_batch_clear(batch);

            // 1. continuation tokens for active streams (parallel.cpp pattern).
            //    Streams admitted in step 2 below are not here yet: their first
            //    token is sampled from the prompt's last-position logits.
            for (auto & st : slots) {
                if (st.req_idx == -1) {
                    continue;
                }
                st.i_batch = batch.n_tokens;
                common_batch_add(batch, st.sampled, st.n_past++, { st.seq_id }, true);
            }

            // 2. admit pending requests, whole, while slots + cells allow
            std::vector<int> admitted;  // req indices admitted this iteration
            while (next_req < reqs.size()) {
                auto & q = reqs[next_req];
                if (q.n_pending == 0) {  // rejected up front
                    next_req++;
                    continue;
                }
                const int need = (int) q.tokens.size() + q.n_samples * q.max_new;
                if (q.n_samples > n_free_slots || n_ctx_used + need > n_ctx - ctx_margin) {
                    break;  // wait for actives to drain
                }

                // init ALL samplers before touching the batch: a failed init must
                // reject the request cleanly, not leave half an admission behind
                std::vector<common_sampler *> smpls(q.n_samples, nullptr);
                bool smpl_ok = true;
                for (int s = 0; s < q.n_samples; ++s) {
                    auto sparams   = params.sampling;
                    sparams.temp   = q.temp;
                    sparams.top_p  = q.top_p;
                    sparams.seed   = (uint32_t) (q.seed_base + s);
                    smpls[s] = common_sampler_init(model, sparams);
                    smpl_ok  = smpl_ok && smpls[s] != nullptr;
                }
                if (!smpl_ok) {
                    for (auto * s : smpls) {
                        if (s) {
                            common_sampler_free(s);
                        }
                    }
                    for (int s = 0; s < q.n_samples; ++s) {
                        emit_error(q, s, "sampler init failed");
                        n_results++;
                    }
                    q.n_pending = 0;
                    next_req++;
                    continue;
                }

                // take slots: first one hosts the prompt, siblings get the KV copy
                std::vector<stream *> take;
                for (auto & st : slots) {
                    if (st.req_idx == -1 && (int) take.size() < q.n_samples) {
                        take.push_back(&st);
                    }
                }
                q.seq_prompt = take[0]->seq_id;
                q.forked = q.n_samples == 1;  // nothing to fork for single-sample requests
                for (size_t i = 0; i < q.tokens.size(); ++i) {
                    common_batch_add(batch, q.tokens[i], (llama_pos) i, { q.seq_prompt },
                                     i + 1 == q.tokens.size());
                }
                q.i_last = batch.n_tokens - 1;
                const int i_last = q.i_last;

                for (int s = 0; s < q.n_samples; ++s) {
                    stream & st = *take[s];
                    st.req_idx    = (int) next_req;
                    st.sample_idx = s;
                    st.n_past     = (int) q.tokens.size();
                    st.n_prompt   = (int) q.tokens.size();
                    st.n_decoded  = 0;
                    st.max_new    = q.max_new;
                    st.i_batch    = i_last;         // all first tokens sample the prompt logits
                    st.response.clear();
                    st.lps.clear();                 // slots are reused across requests
                    st.tlens.clear();
                    st.lp_sum     = 0.0;
                    st.lp_eos     = 0.0f;
                    if (st.smpl) {
                        common_sampler_free(st.smpl);
                    }
                    st.smpl = smpls[s];
                }

                q.n_active  = q.n_samples;
                q.n_pending = 0;
                n_free_slots -= q.n_samples;
                n_ctx_used   += need;
                admitted.push_back((int) next_req);
                next_req++;
            }

            if (batch.n_tokens == 0) {
                break;  // group drained
            }

            // 3. decode in chunks of n_batch, halving on cache pressure; sample
            //    INSIDE the chunk loop (parallel.cpp): logits live only until the
            //    next llama_decode, and common_sampler_sample indexes into the
            //    LAST decoded view — absolute batch indices crash on multi-chunk
            //    groups (get_logits_ith assert; found by the 27-row budget test)
            int32_t n_batch = params.n_batch;
            bool decode_failed = false;
            for (int32_t i = 0; i < batch.n_tokens; ) {
                const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

                llama_batch batch_view = {
                    n_tokens,
                    batch.token    + i,
                    nullptr,
                    batch.pos      + i,
                    batch.n_seq_id + i,
                    batch.seq_id   + i,
                    batch.logits   + i,
                };

                const int ret = llama_decode(ctx, batch_view);
                if (ret != 0) {
                    if (n_batch == 1 || ret < 0) {
                        fprintf(stderr, "iolbatch: llama_decode failed, ret=%d\n", ret);
                        decode_failed = true;
                        break;
                    }
                    n_batch /= 2;  // retry smaller to find free KV cells
                    continue;
                }

                // 3a. fork freshly decoded prompts to their sibling sequences BEFORE
                //     sampling this view: if sample 0 finishes instantly (EOS first),
                //     its seq_rm below must not take the un-forked prompt cells with it
                for (int r : admitted) {
                    auto & q = reqs[r];
                    if (q.i_last < i || q.i_last >= i + n_tokens || q.forked) {
                        continue;
                    }
                    for (auto & st : slots) {
                        if (st.req_idx == r && st.seq_id != q.seq_prompt) {
                            llama_memory_seq_cp(mem, q.seq_prompt, st.seq_id, 0, (llama_pos) q.tokens.size());
                        }
                    }
                    q.forked = true;
                }

                // 3b. sample every stream whose logits sit in this view
                for (auto & st : slots) {
                    if (st.req_idx == -1 || st.i_batch < i || st.i_batch >= i + n_tokens) {
                        continue;
                    }
                    request & q = reqs[st.req_idx];

                    const int i_logits = st.i_batch - i;  // read BEFORE clearing i_batch
                    const llama_token tok = common_sampler_sample(st.smpl, ctx, i_logits);
                    common_sampler_accept(st.smpl, tok, true);
                    st.sampled = tok;
                    st.i_batch = -1;
                    st.n_decoded += 1;

                    const bool is_eog = llama_vocab_is_eog(vocab, tok);
                    // logits stay valid only until the next llama_decode, and
                    // llama_get_logits_ith indexes the LAST decoded view — same
                    // contract common_sampler_sample above just used
                    const float lp = q.logprobs
                                         ? token_logprob(llama_get_logits_ith(ctx, i_logits), n_vocab, tok)
                                         : 0.0f;
                    if (!is_eog) {
                        const std::string piece = common_token_to_piece(ctx, tok);
                        st.response += piece;
                        if (q.logprobs) {
                            st.lps.push_back(lp);
                            st.tlens.push_back((int) piece.size());
                            st.lp_sum += lp;
                        }
                    } else if (q.logprobs) {
                        st.lp_eos = lp;
                    }

                    if (is_eog || st.n_decoded >= st.max_new) {
                        emit_result(q, st, is_eog ? "eos" : "limit");
                        n_results++;

                        llama_memory_seq_rm(mem, st.seq_id, -1, -1);
                        n_ctx_used -= st.max_new;
                        n_free_slots += 1;
                        q.n_active -= 1;
                        if (q.n_active == 0) {
                            n_ctx_used -= (int) q.tokens.size();  // prompt cells freed with last sibling
                        }
                        st.req_idx = -1;
                    }
                }

                i += n_tokens;
                n_batch = params.n_batch;
            }
            if (decode_failed) {
                // salvage whatever the active streams already produced
                for (auto & st : slots) {
                    if (st.req_idx != -1) {
                        emit_result(reqs[st.req_idx], st, "error");
                        n_results++;
                        st.req_idx = -1;
                    }
                }
                for (size_t r = next_req; r < reqs.size(); ++r) {
                    for (int s = 0; s < reqs[r].n_samples && reqs[r].n_pending; ++s) {
                        emit_error(reqs[r], s, "decode failed before admission");
                        n_results++;
                    }
                }
                fatal = true;
                break;
            }
        }

        // hygiene: nothing survives a group
        llama_memory_clear(mem, true);
        for (auto & st : slots) {
            st.req_idx = -1;
            st.i_batch = -1;
        }

        proto_emit(json{
            {"group", g_group},
            {"done", true},
            {"n_results", n_results},
            {"t_ms", (int) ((ggml_time_us() - t_group_start) / 1000)},
        });
    }

    for (auto & st : slots) {
        if (st.smpl) {
            common_sampler_free(st.smpl);
        }
    }
    llama_batch_free(batch);
    llama_backend_free();
    fclose(g_proto);

    return 0;
}
