import { TypeSafeClient } from "@typesafe-ai/sdk";

export function choiceClient(answer: string, probabilities: Record<string, number>, confidence = 0.95): TypeSafeClient {
  return sequenceChoiceClient([{ answer, probabilities, confidence }]);
}

export function answersClient(answers: Record<string, unknown>, inputTokens = 17): TypeSafeClient {
  return new TypeSafeClient({
    apiKey: "test-key",
    baseURL: "https://jev.invalid",
    retry: { maxRetries: 0 },
    fetch: async () =>
      new Response(
        JSON.stringify({
          model: "jev-test",
          answers,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
  });
}

export function sequenceChoiceClient(
  responses: { answer: string; probabilities: Record<string, number>; confidence?: number }[],
): TypeSafeClient {
  let call = 0;
  return new TypeSafeClient({
    apiKey: "test-key",
    baseURL: "https://jev.invalid",
    retry: { maxRetries: 0 },
    fetch: async () => {
      const response = responses[call];
      call += 1;
      if (!response) throw new Error(`Unexpected fake Jev call ${call}`);
      return new Response(
        JSON.stringify({
          model: "jev-test",
          answers: {
            answer: {
              type: "choice",
              choice: response.answer,
              confidence: response.confidence ?? 0.95,
              probabilities: response.probabilities,
            },
          },
          usage: { input_tokens: 17, output_tokens: 0 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    },
  });
}

export function failingClient(): TypeSafeClient {
  return new TypeSafeClient({
    apiKey: "test-key",
    baseURL: "https://jev.invalid",
    retry: { maxRetries: 0 },
    fetch: async () => {
      throw new Error("Jev unavailable");
    },
  });
}

export function timeoutClient(): TypeSafeClient {
  return new TypeSafeClient({
    apiKey: "test-key",
    baseURL: "https://jev.invalid",
    retry: { maxRetries: 0 },
    timeout: 5,
    fetch: async (_input, init) =>
      new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        if (signal?.aborted) {
          reject(signal.reason);
          return;
        }
        signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
      }),
  });
}
