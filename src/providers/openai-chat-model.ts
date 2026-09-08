import { fail, tryPromise } from "effect/Effect";
import type { Layer } from "effect/Layer";
import { succeed as layerSucceed } from "effect/Layer";

import {
  MessageRole,
  ModelError,
  type ModelMessage,
  type ModelOutput,
} from "../harness/core";
import type { GenerateConfig, ModelService } from "../harness/model";
import { Model } from "../harness/model";

export interface OpenAIChatModelConfig {
  readonly model: string;
  readonly apiKey: string;
  readonly baseUrl: string;
}

/** OpenAI-compatible root including `/v1` (vLLM: http://127.0.0.1:8000/v1). */
export function normalizeOpenAIChatBaseUrl(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/u, "");
  if (trimmed.endsWith("/v1")) {
    return trimmed;
  }
  return `${trimmed}/v1`;
}

export function localOpenAIChatBaseUrlFromEnv(
  env: Readonly<Record<string, string | undefined>> = process.env
): string | undefined {
  const localRoot = env["VLLM_BASE_URL"] ?? env["OPENAI_BASE_URL"];
  if (localRoot === undefined || localRoot.length === 0) {
    return undefined;
  }
  return localRoot;
}

function chatMessages(
  messages: readonly ModelMessage[]
): { role: string; content: string }[] {
  return messages.map((m) => ({
    role: m.role,
    content: m.content,
  }));
}

export function makeOpenAIChatModelLayer(
  config: OpenAIChatModelConfig
): Layer<Model> {
  const baseUrl = normalizeOpenAIChatBaseUrl(config.baseUrl);
  const service: ModelService = {
    generate: (messages, generateConfig: GenerateConfig) => {
      if (
        generateConfig.tools !== undefined &&
        generateConfig.tools.length > 0
      ) {
        return fail(
          new ModelError({
            message:
              "Local vLLM chat backend does not implement tool calls; unset tools for gpqa_diamond.",
          })
        );
      }
      return tryPromise({
        try: async () => {
          const started = Date.now();
          const body: Record<string, unknown> = {
            model: config.model,
            messages: chatMessages(messages),
            temperature: generateConfig.temperature,
            max_tokens: generateConfig.maxTokens ?? 8192,
          };
          const resp = await fetch(`${baseUrl}/chat/completions`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${config.apiKey}`,
            },
            body: JSON.stringify(body),
          });
          const text = await resp.text();
          if (!resp.ok) {
            throw new ModelError({
              status: resp.status,
              message: `vLLM HTTP ${resp.status}: ${text.slice(0, 2000)}`,
            });
          }
          const json = JSON.parse(text) as {
            choices?: { message?: { content?: string } }[];
          };
          const content = json.choices?.[0]?.message?.content ?? "";
          const output: ModelOutput = {
            completion: content,
            message: { role: MessageRole.Assistant, content },
            generationTimeMs: Date.now() - started,
          };
          return output;
        },
        catch: (e) =>
          e instanceof ModelError
            ? e
            : new ModelError({
                message: e instanceof Error ? e.message : String(e),
              }),
      });
    },
  };
  return layerSucceed(Model, Model.of(service));
}
