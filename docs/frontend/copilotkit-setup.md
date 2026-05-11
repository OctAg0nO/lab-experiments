# CopilotKit + A2UI Frontend Setup Guide

This guide sets up a React frontend that connects to the OctAg0nO LiveKit agent,
renders A2UI components, and supports AG-UI features (shared state, frontend tools,
human-in-the-loop).

## Prerequisites

- Node.js 18+
- LiveKit server running (`docker run --rm -p 7880:7880 livekit/livekit-server`)
- OctAg0nO agent running (`uv run python -m lab.15_ray_sglang livekit-worker`)

## Setup

```bash
# Create a new Next.js app with CopilotKit
npx create-ag-ui-app my-octagono-app
cd my-octagono-app

# Install LiveKit React SDK
npm install @livekit/components-react @livekit/components-styles
```

## Configuration

### 1. LiveKit + CopilotKit Integration

```tsx
// app/page.tsx
'use client';

import { LiveKitRoom, VideoConference } from '@livekit/components-react';
import { CopilotKit } from '@copilotkit/react-core';
import { CopilotSidebar } from '@copilotkit/react-ui';
import { useA2UI } from '@a2ui/react';

export default function Page() {
  const token = useLiveKitToken(); // your token logic

  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <LiveKitRoom token={token} serverUrl={process.env.NEXT_PUBLIC_LK_URL}>
        <A2UISurface />
        <CopilotSidebar />
        <VoiceControls />
      </LiveKitRoom>
    </CopilotKit>
  );
}
```

### 2. A2UI Surface Renderer

The A2UI Surface listens on the LiveKit data channel topic `a2ui`
and renders components using CopilotKit's generative UI system:

```tsx
// components/A2UISurface.tsx
'use client';

import { useRoom } from '@livekit/components-react';
import { useEffect, useState } from 'react';

interface A2UIComponent {
  id: string;
  component: string;
  props: Record<string, any>;
}

export function A2UISurface() {
  const room = useRoom();
  const [components, setComponents] = useState<Record<string, A2UIComponent>>({});

  useEffect(() => {
    if (!room) return;

    const handleData = (packet: any) => {
      try {
        const msg = JSON.parse(new TextDecoder().decode(packet.data));
        if (msg.version !== 'v0.10') return;

        if (msg.createSurface) {
          // Initialize surface — store theme/catalog
          console.log('Surface created:', msg.createSurface.surfaceId);
        } else if (msg.updateComponents) {
          // Add or update components
          const updates: Record<string, A2UIComponent> = {};
          for (const comp of msg.updateComponents.components) {
            updates[comp.id] = comp;
          }
          setComponents(prev => ({ ...prev, ...updates }));
        } else if (msg.deleteSurface) {
          // Remove surface and its components
          setComponents({});
        }
      } catch (e) {
        // ignore parse errors
      }
    };

    room.on('dataReceived', handleData);
    return () => { room.off('dataReceived', handleData); };
  }, [room]);

  return (
    <div className="a2ui-surface">
      {Object.values(components).map(comp => (
        <A2UIComponentRenderer key={comp.id} component={comp} />
      ))}
    </div>
  );
}
```

### 3. AG-UI Shared State

```tsx
// hooks/useSharedState.ts
export function useSharedState() {
  const room = useRoom();

  const updateState = (path: string, value: any) => {
    room?.localParticipant.publishData(
      new TextEncoder().encode(JSON.stringify({
        type: 'shared_state_update',
        data: { path, value },
      })),
      topic: 'agui',
    );
  };

  return { updateState };
}
```

## Running

```bash
npm run dev
# Opens at http://localhost:3000
```

The frontend connects to LiveKit, joins the voice session,
and renders A2UI components sent by the OctAg0nO agent.
