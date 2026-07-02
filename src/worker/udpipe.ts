import { parseConllu, type Token } from "./disambiguation";

const UDPIPE_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process";
const UDPIPE_MODEL = "lithuanian-alksnis";
const UDPIPE_TIMEOUT_MS = 10_000;

type UdpipeResponse = {
  result?: unknown;
};

export async function tagText(text: string): Promise<Token[]> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), UDPIPE_TIMEOUT_MS);

  try {
    const body = new URLSearchParams({
      tokenizer: "",
      tagger: "",
      model: UDPIPE_MODEL,
      data: text,
    });

    const response = await fetch(UDPIPE_URL, {
      method: "POST",
      headers: {
        "content-type": "application/x-www-form-urlencoded",
        accept: "application/json",
      },
      body,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error("UDPipe response was not ok.");
    }

    const payload = (await response.json()) as UdpipeResponse;
    if (typeof payload.result !== "string") {
      throw new Error("UDPipe response did not include CoNLL-U.");
    }

    return parseConllu(payload.result);
  } finally {
    clearTimeout(timeout);
  }
}
