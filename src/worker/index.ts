import type {
  AccentRequest,
  AccentResponse,
  ErrorResponse,
  WordResponse,
} from "../shared/types";
import {
  accentText,
  lookupWordVariantsKV,
  UpstreamError,
  WORD_CACHE_SECONDS,
} from "./vdu";
import { toPublicVariants } from "./disambiguation";

const MAX_TEXT_LENGTH = 20_000;

export interface Env {
  ASSETS: Fetcher;
  WORDS: KVNamespace;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    try {
      if (url.pathname === "/api/accent") {
        return handleAccent(request, env, ctx);
      }

      if (url.pathname === "/api/word") {
        return handleWord(request, url, env, ctx);
      }

      if (url.pathname.startsWith("/api/")) {
        return json<ErrorResponse>({ error: "API maršrutas nerastas." }, 404);
      }

      return env.ASSETS.fetch(request);
    } catch (error) {
      if (error instanceof UpstreamError) {
        return json<ErrorResponse>({ error: error.message }, 502);
      }

      console.error(error);
      return json<ErrorResponse>({ error: "Įvyko netikėta klaida." }, 500);
    }
  },
} satisfies ExportedHandler<Env>;

async function handleAccent(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  if (request.method !== "POST") {
    return json<ErrorResponse>({ error: "Metodas nepalaikomas." }, 405);
  }

  const payload = await readJson<AccentRequest>(request);
  if (typeof payload?.text !== "string" || payload.text.trim().length === 0) {
    return json<ErrorResponse>({ error: "Įveskite tekstą." }, 400);
  }

  if (payload.text.length > MAX_TEXT_LENGTH) {
    return json<ErrorResponse>({ error: "Tekstas per ilgas." }, 413);
  }

  const response = await accentText(payload.text, {
    lookupVariants: (word) => lookupWordVariantsKV(word, env, ctx),
  });
  return json<AccentResponse>(response);
}

async function handleWord(
  request: Request,
  url: URL,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  if (request.method !== "GET") {
    return json<ErrorResponse>({ error: "Metodas nepalaikomas." }, 405);
  }

  const word = url.searchParams.get("w")?.trim() ?? "";
  if (!word) {
    return json<ErrorResponse>({ error: "Trūksta žodžio." }, 400);
  }

  const variants = await lookupWordVariantsKV(word, env, ctx);
  return json<WordResponse>(
    { variants: toPublicVariants(variants) },
    200,
    { "cache-control": `public, max-age=${WORD_CACHE_SECONDS}` },
  );
}

async function readJson<T>(request: Request): Promise<T | null> {
  try {
    return (await request.json()) as T;
  } catch {
    return null;
  }
}

function json<T>(
  body: T,
  status = 200,
  headers: HeadersInit = {},
): Response {
  return Response.json(body, {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "x-content-type-options": "nosniff",
      ...headers,
    },
  });
}
