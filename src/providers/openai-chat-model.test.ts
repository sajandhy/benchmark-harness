import { afterEach, describe, expect, it } from "bun:test";
import { gen, provide, runPromiseExit } from "effect/Effect";

import { assertSuccess } from "../../test/helpers/exit-asserts";
import { Model } from "../harness/model";
import {
  localOpenAIChatBaseUrlFromEnv,
  makeOpenAIChatModelLayer,
  normalizeOpenAIChatBaseUrl,
} from "./openai-chat-model";

describe("openai-chat-model", () => {
  let restore: (() => void) | undefined;

  afterEach(() => {
    restore?.();
    restore = undefined;
  });

  it("normalizes vLLM roots to /v1", () => {
    expect(normalizeOpenAIChatBaseUrl("http://127.0.0.1:8000")).toBe(
      "http://127.0.0.1:8000/v1"
    );
    expect(normalizeOpenAIChatBaseUrl("http://127.0.0.1:8000/v1/")).toBe(
      "http://127.0.0.1:8000/v1"
    );
  });

  it("reads VLLM_BASE_URL from env", () => {
    expect(localOpenAIChatBaseUrlFromEnv({})).toBeUndefined();
    expect(
      localOpenAIChatBaseUrlFromEnv({ VLLM_BASE_URL: "http://127.0.0.1:8000/v1" })
    ).toBe("http://127.0.0.1:8000/v1");
  });

  it("posts chat completions and returns the assistant text", async () => {
    const originalFetch = globalThis.fetch;
    let request: Request | undefined;
    globalThis.fetch = async (input, init) => {
      request = input instanceof Request ? input : new Request(input, init);
      return new Response(
        JSON.stringify({
          choices: [{ message: { content: "Answer: B" } }],
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    };
    restore = () => {
      globalThis.fetch = originalFetch;
    };

    const exit = await runPromiseExit(
      gen(function* () {
        const model = yield* Model;
        return yield* model.generate([{ role: "user", content: "question" }], {
          temperature: 0.5,
          reasoningEffort: "none",
        });
      }).pipe(
        provide(
          makeOpenAIChatModelLayer({
            model: "amd/GLM-5.3-Quark-MXFP4-AttnFP8",
            apiKey: "EMPTY",
            baseUrl: "http://127.0.0.1:8000",
          })
        )
      )
    );
    assertSuccess(exit);
    expect(request?.url).toBe("http://127.0.0.1:8000/v1/chat/completions");
    expect(request?.signal).toBeDefined();
    const body: unknown = JSON.parse(await request!.clone().text());
    expect(body).toMatchObject({
      model: "amd/GLM-5.3-Quark-MXFP4-AttnFP8",
      messages: [{ role: "user", content: "question" }],
      temperature: 0.5,
    });
    expect(body).not.toHaveProperty("max_tokens");
    expect(exit.value.completion).toBe("Answer: B");
  });
});
