import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { initChatModel } from "langchain/chat_models/universal";

/** Error raised when a model identifier is not supported by the registry. */
export class UnsupportedModelError extends Error {
  /** Creates an error containing the unsupported model identifier. */
  constructor(readonly model: string) {
    super(`Unsupported model "${model}". Expected openai:* or anthropic:* identifiers.`);
    this.name = "UnsupportedModelError";
  }
}

export type ModelResolver = (modelId: string) => Promise<BaseChatModel>;

/**
 * Resolves provider-prefixed model identifiers through LangChain's universal
 * chat model initializer.
 */
export async function resolveModel(modelId: string): Promise<BaseChatModel> {
  if (!modelId.startsWith("openai:") && !modelId.startsWith("anthropic:")) {
    throw new UnsupportedModelError(modelId);
  }
  return initChatModel(modelId);
}
