export type LocalDownloadGateState =
  | "inactive"
  | "checking-cache"
  | "needs-consent"
  | "needs-redownload"
  | "loading"
  | "ready"
  | "failed";

export type LocalDownloadGateDeps = {
  hasCachedModel: () => Promise<boolean>;
  ensureEngine: () => Promise<unknown>;
  isEngineReady?: () => boolean;
  wasPreviouslyReady?: () => boolean;
  markReady?: () => void;
  onState?: (state: LocalDownloadGateState) => void;
};

export type LocalDownloadGate = {
  readonly state: LocalDownloadGateState;
  enterLocalMode: () => Promise<LocalDownloadGateState>;
  consentToDownload: () => Promise<LocalDownloadGateState>;
  leaveLocalMode: () => void;
};

export function createLocalDownloadGate(
  deps: LocalDownloadGateDeps,
): LocalDownloadGate {
  let state: LocalDownloadGateState = "inactive";
  let runId = 0;

  const setState = (next: LocalDownloadGateState) => {
    if (state === next) {
      return;
    }
    state = next;
    deps.onState?.(state);
  };

  const loadEngine = async (currentRunId: number) => {
    setState("loading");
    try {
      await deps.ensureEngine();
      if (currentRunId === runId) {
        deps.markReady?.();
        setState("ready");
      }
    } catch (error) {
      if (currentRunId === runId) {
        setState("failed");
      }
      throw error;
    }
    return state;
  };

  return {
    get state() {
      return state;
    },

    async enterLocalMode() {
      const currentRunId = ++runId;
      setState("checking-cache");
      const cached = await deps.hasCachedModel();
      if (currentRunId !== runId) {
        return state;
      }

      if (!cached) {
        setState(deps.wasPreviouslyReady?.() ? "needs-redownload" : "needs-consent");
        return state;
      }

      if (deps.isEngineReady?.()) {
        deps.markReady?.();
        setState("ready");
        return state;
      }

      return loadEngine(currentRunId);
    },

    async consentToDownload() {
      const currentRunId = ++runId;
      if (deps.isEngineReady?.()) {
        deps.markReady?.();
        setState("ready");
        return state;
      }

      return loadEngine(currentRunId);
    },

    leaveLocalMode() {
      runId += 1;
      setState("inactive");
    },
  };
}
