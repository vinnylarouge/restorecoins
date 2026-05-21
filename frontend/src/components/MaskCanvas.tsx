import { useEffect, useMemo, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Line } from "react-konva";
import useImage from "use-image";
import { base64ToDataUrl } from "../api/client";

// Painted mask is composed in an off-screen canvas — Konva handles UI strokes,
// but for the backend we need a clean grayscale PNG matching the input size.

interface Stroke {
  points: number[];        // [x0,y0,x1,y1,...] in image coordinates
  mode: "paint" | "erase";
  size: number;
}

const MAX_W = 720;

export function MaskCanvas({
  imageDataUrl, proposedMaskB64, threshold,
  brushSize, mode, onMaskChange,
}: {
  imageDataUrl: string;
  proposedMaskB64?: string;
  threshold: number;
  brushSize: number;
  mode: "paint" | "erase";
  onMaskChange: (b64: string) => void;
}) {
  const [img] = useImage(imageDataUrl);
  const proposedDataUrl = useMemo(
    () => proposedMaskB64 ? base64ToDataUrl(proposedMaskB64) : undefined,
    [proposedMaskB64],
  );
  const [proposedImg] = useImage(proposedDataUrl ?? "");
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const drawing = useRef(false);

  const { displayW, displayH, scale } = useMemo(() => {
    if (!img) return { displayW: 0, displayH: 0, scale: 1 };
    const s = Math.min(1, MAX_W / img.width);
    return { displayW: img.width * s, displayH: img.height * s, scale: s };
  }, [img]);

  // Whenever strokes/threshold/proposed changes, recompose the mask PNG.
  useEffect(() => {
    if (!img) return;
    const cnv = document.createElement("canvas");
    cnv.width = img.width;
    cnv.height = img.height;
    const ctx = cnv.getContext("2d")!;
    // Black background; we'll paint white where pixels should be inpainted.
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, cnv.width, cnv.height);

    // 1. Bake the proposed map (thresholded) into the canvas as the base layer.
    if (proposedImg) {
      const tmp = document.createElement("canvas");
      tmp.width = img.width; tmp.height = img.height;
      const tctx = tmp.getContext("2d")!;
      tctx.drawImage(proposedImg, 0, 0, img.width, img.height);
      const tdata = tctx.getImageData(0, 0, img.width, img.height);
      const out = ctx.getImageData(0, 0, img.width, img.height);
      for (let i = 0; i < tdata.data.length; i += 4) {
        const lum = tdata.data[i];
        const v = lum >= threshold ? 255 : 0;
        out.data[i] = v; out.data[i + 1] = v; out.data[i + 2] = v; out.data[i + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
    }

    // 2. Apply user strokes in image coordinates.
    for (const stroke of strokes) {
      ctx.lineWidth = stroke.size / scale;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = stroke.mode === "paint" ? "white" : "black";
      ctx.globalCompositeOperation = "source-over";
      ctx.beginPath();
      ctx.moveTo(stroke.points[0] / scale, stroke.points[1] / scale);
      for (let k = 2; k < stroke.points.length; k += 2) {
        ctx.lineTo(stroke.points[k] / scale, stroke.points[k + 1] / scale);
      }
      ctx.stroke();
    }

    cnv.toBlob((blob) => {
      if (!blob) return;
      const reader = new FileReader();
      reader.onload = () => {
        const s = reader.result as string;
        onMaskChange(s.slice(s.indexOf(",") + 1));
      };
      reader.readAsDataURL(blob);
    }, "image/png");
  }, [img, proposedImg, threshold, strokes, scale, onMaskChange]);

  if (!img) return <div className="text-ink/50">loading image…</div>;

  return (
    <Stage
      width={displayW} height={displayH}
      style={{ background: "rgba(0,0,0,0.04)", touchAction: "none" }}
      onMouseDown={(e) => {
        drawing.current = true;
        const p = e.target.getStage()!.getPointerPosition()!;
        setStrokes((s) => [...s, { mode, size: brushSize, points: [p.x, p.y] }]);
      }}
      onMouseMove={(e) => {
        if (!drawing.current) return;
        const p = e.target.getStage()!.getPointerPosition()!;
        setStrokes((s) => {
          const copy = [...s];
          copy[copy.length - 1] = {
            ...copy[copy.length - 1],
            points: [...copy[copy.length - 1].points, p.x, p.y],
          };
          return copy;
        });
      }}
      onMouseUp={() => { drawing.current = false; }}
      onMouseLeave={() => { drawing.current = false; }}
    >
      <Layer>
        <KonvaImage image={img} width={displayW} height={displayH} />
      </Layer>
      {proposedImg && (
        <Layer opacity={0.35} listening={false}>
          {/* Visualises the thresholded proposal as a tinted overlay so the user
              can see what the server suggested before they paint over it. */}
          <KonvaImage image={proposedImg} width={displayW} height={displayH}
                      globalCompositeOperation="multiply" />
        </Layer>
      )}
      <Layer>
        {strokes.map((stroke, i) => (
          <Line
            key={i}
            points={stroke.points}
            stroke={stroke.mode === "paint" ? "rgba(58,90,58,0.5)" : "rgba(244,237,224,0.85)"}
            strokeWidth={stroke.size}
            lineCap="round" lineJoin="round"
            tension={0.3}
            globalCompositeOperation={stroke.mode === "erase" ? "destination-out" : "source-over"}
          />
        ))}
      </Layer>
    </Stage>
  );
}
