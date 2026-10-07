"use client";

import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { predictPrice, PredictionError, serviceIsUp, type FailureKind, type Prediction } from "@/lib/api";
import { prepareImage, uploadName } from "@/lib/image";
import styles from "./PricePredictor.module.css";

type State =
  | { step: "choose" }
  | { step: "working"; preview: string }
  | { step: "done"; preview: string; result: Prediction }
  | { step: "failed"; preview: string; message: string; file: File; retryable: boolean };

// Failures where sending the same photo again can help.
const RETRYABLE: FailureKind[] = ["unavailable", "offline", "timeout"];

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

// Where the estimate sits inside its range, 0 to 1. On a log scale, because
// the range is "x0.4 to x2.8 of the estimate": in dollars the estimate sits
// near the low end, which would make the bar look lopsided.
function positionInRange({ price, range: { low, high } }: Prediction): number {
  if (high <= low) return 0.5;
  const t = (Math.log(price) - Math.log(low)) / (Math.log(high) - Math.log(low));
  return Math.min(1, Math.max(0, t));
}

export function PricePredictor() {
  const [state, setState] = useState<State>({ step: "choose" });
  const [serviceDown, setServiceDown] = useState(false);
  const [dragging, setDragging] = useState(false);
  const previewUrl = useRef<string | null>(null);
  const uploaded = useRef(false);

  useEffect(() => {
    // If an upload has finished by the time this first check answers, the
    // upload's outcome is the newer news, so the check doesn't override it.
    serviceIsUp().then((up) => {
      if (!uploaded.current) setServiceDown(!up);
    });
    return () => {
      if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    };
  }, []);

  const estimate = useCallback(async (file: File) => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    const preview = URL.createObjectURL(file);
    previewUrl.current = preview;
    setState({ step: "working", preview });
    try {
      const image = await prepareImage(file);
      const result = await predictPrice(image, uploadName(file, image));
      uploaded.current = true;
      setServiceDown(false);
      setState({ step: "done", preview, result });
    } catch (err) {
      uploaded.current = true;
      const known = err instanceof PredictionError;
      if (known && err.kind === "unavailable") setServiceDown(true);
      setState({
        step: "failed",
        preview,
        file,
        message: known ? err.message : "Something went wrong. Try again.",
        retryable: !known || RETRYABLE.includes(err.kind),
      });
    }
  }, []);

  // Paste a copied image anywhere on the page (handy on desktop).
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (file) estimate(file);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [estimate]);

  const onChosen = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // so picking the same photo again still counts
    if (file) estimate(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files);
    const file = files.find((f) => f.type.startsWith("image/")) ?? files[0];
    if (file) estimate(file);
  };

  return (
    <section className={styles.card} aria-busy={state.step === "working"}>
      {serviceDown && (
        <div className={styles.banner} role="status">
          The price service isn&apos;t reachable right now.{" "}
          <button className={styles.linkButton} onClick={() => serviceIsUp().then((up) => setServiceDown(!up))}>
            Check again
          </button>
        </div>
      )}

      {state.step === "choose" ? (
        <div
          className={`${styles.dropZone} ${dragging ? styles.dragging : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <p className={styles.dropTitle}>Add a product photo</p>
          {/* Upright matters: lying on its side, a product can get twice the price (Phase 6). */}
          <p className={styles.hint}>One item, upright, in good light, works best.</p>
          <div className={styles.actions}>
            {/* Labels open the file inputs natively, which works everywhere, iOS included. */}
            <label className={`${styles.button} ${styles.cameraButton}`}>
              Take a photo
              <input className={styles.fileInput} type="file" accept="image/*" capture="environment" onChange={onChosen} />
            </label>
            <label className={styles.button}>
              Choose a photo
              <input className={styles.fileInput} type="file" accept="image/*" onChange={onChosen} />
            </label>
          </div>
          <p className={`${styles.hint} ${styles.desktopOnly}`}>or drop a photo here, or paste one</p>
        </div>
      ) : (
        <div className={styles.layout}>
          {/* A blob: URL of the user's own photo; next/image optimisation doesn't apply. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className={styles.preview} src={state.preview} alt="The photo being priced" />

          <div className={styles.panel} aria-live="polite">
            {state.step === "working" && (
              <p className={styles.working}>
                <span className={styles.spinner} aria-hidden="true" />
                Estimating the price…
              </p>
            )}
            {state.step === "done" && <Result result={state.result} />}
            {state.step === "failed" && (
              <p className={styles.error} role="alert">
                {state.message}
              </p>
            )}

            {state.step !== "working" && (
              <div className={styles.actions}>
                {state.step === "failed" && state.retryable && (
                  <button className={styles.button} onClick={() => estimate(state.file)}>
                    Try again
                  </button>
                )}
                <button
                  className={state.step === "failed" && state.retryable ? styles.secondaryButton : styles.button}
                  onClick={() => setState({ step: "choose" })}
                >
                  {state.step === "done" ? "Price another photo" : "Choose another photo"}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

function Result({ result }: { result: Prediction }) {
  const { low, high, coverage } = result.range;
  return (
    <div className={styles.result}>
      <p className={styles.label}>Estimated price</p>
      <p className={styles.price}>{usd.format(result.price)}</p>

      <div className={styles.range} aria-hidden="true">
        <div className={styles.track}>
          <span className={styles.marker} style={{ left: `${positionInRange(result) * 100}%` }} />
        </div>
        <div className={styles.ends}>
          <span>{usd.format(low)}</span>
          <span>{usd.format(high)}</span>
        </div>
      </div>

      <p className={styles.explain}>
        Likely between <strong>{usd.format(low)}</strong> and <strong>{usd.format(high)}</strong>: about{" "}
        {Math.round(coverage * 10)} in 10 products that got a similar estimate were priced in that range.
      </p>
      <p className={styles.caveat}>
        This is estimated from the photo alone. For a multipack or a bulk size the real price is usually
        higher, because the quantity is hard to see in a photo.
      </p>
      <p className={styles.version}>Model {result.model_version}</p>
    </div>
  );
}
