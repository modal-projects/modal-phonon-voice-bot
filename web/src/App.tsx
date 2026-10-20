import { RTVIEvent, type PipecatClient } from "@pipecat-ai/client-js";
import { ConsoleTemplate, ThemeProvider } from "@pipecat-ai/voice-ui-kit";

import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "@pipecat-ai/voice-ui-kit/styles";

const startBotParams = { endpoint: "/api/session" };
const clientOptions = { enableMic: true, enableCam: false };

function offerParams() {
  return {
    webrtcRequestParams: {
      endpoint: `${window.location.origin}/api/offer`,
    },
  };
}

function bindSession(client: PipecatClient) {
  const audio = new Audio();
  audio.autoplay = true;
  // The connect click is the user gesture. Play once here so a later
  // srcObject assignment is allowed to start.
  const unlock = () => {
    audio.play().catch(() => {});
  };
  document.addEventListener("pointerdown", unlock, { once: true });

  const playReceivers = () => {
    const pc = (
      client.transport as { pc?: RTCPeerConnection } | undefined
    )?.pc;
    if (!pc) return;
    const tracks = pc
      .getReceivers()
      .map((receiver) => receiver.track)
      .filter((track): track is MediaStreamTrack => track?.kind === "audio");
    if (!tracks.length) return;
    const current = audio.srcObject as MediaStream | null;
    if (current?.getAudioTracks()[0]?.id === tracks[0].id) return;
    audio.srcObject = new MediaStream(tracks);
    audio.play().catch((error: unknown) => {
      console.warn("bot audio", error);
    });
    // The UI kit plays the track itself once it unmutes. Hand off so the
    // same speech is not heard twice.
    for (const track of tracks) {
      track.addEventListener(
        "unmute",
        () => {
          audio.srcObject = null;
        },
        { once: true },
      );
    }
  };

  let keepalive: number | undefined;
  const stopKeepalive = () => {
    if (keepalive !== undefined) window.clearInterval(keepalive);
    keepalive = undefined;
  };

  client.on(RTVIEvent.Connected, () => {
    const pc = (
      client.transport as { pc?: RTCPeerConnection } | undefined
    )?.pc;
    pc?.addEventListener("track", playReceivers);
    playReceivers();
    stopKeepalive();
    keepalive = window.setInterval(() => {
      fetch("/api/keepalive").catch(() => {});
    }, 30000);
  });
  client.on(RTVIEvent.Disconnected, () => {
    stopKeepalive();
    fetch("/api/session", { method: "DELETE" }).catch(() => {});
  });
}

export default function App() {
  return (
    <ThemeProvider>
      <div className="w-full h-dvh bg-background">
        <ConsoleTemplate
          transportType="smallwebrtc"
          titleText="Phonon"
          clientOptions={clientOptions}
          startBotParams={startBotParams}
          startBotResponseTransformer={offerParams}
          onClient={bindSession}
          noUserVideo
          noBotVideo
          noScreenControl
          noDTMF
          noTextInput
          textRenderMode="instant"
        />
      </div>
    </ThemeProvider>
  );
}
