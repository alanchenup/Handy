import React, { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import { Film, Link2 } from "lucide-react";
import { toast } from "sonner";
import { commands } from "@/bindings";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { useSettings } from "@/hooks/useSettings";

export const MediaTranscribeSettings: React.FC = () => {
  const { t } = useTranslation();
  const { settings } = useSettings();
  const [localPath, setLocalPath] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const [applyVad, setApplyVad] = useState(true);
  const [busy, setBusy] = useState(false);
  const [resultText, setResultText] = useState("");

  const postProcess = settings?.post_process_enabled ?? false;

  const effectiveSource = localPath ?? url.trim();
  const canTranscribe = effectiveSource.length > 0 && !busy;

  const runTranscription = useCallback(async () => {
    if (!effectiveSource) {
      toast.error(t("mediaTranscribe.errors.noInput"));
      return;
    }
    setBusy(true);
    setResultText("");
    try {
      const res = await commands.transcribeMediaSource(
        effectiveSource,
        applyVad,
        postProcess,
      );
      if (res.status === "error") {
        toast.error(res.error);
        return;
      }
      setResultText(res.data.text);
      toast.success(t("mediaTranscribe.done"));
    } catch (e) {
      toast.error(
        e instanceof Error ? e.message : t("mediaTranscribe.errors.failed"),
      );
    } finally {
      setBusy(false);
    }
  }, [effectiveSource, applyVad, postProcess, t]);

  const onPickFile = useCallback(async () => {
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [
        {
          name: t("mediaTranscribe.fileFilterName"),
          extensions: [
            "mp4",
            "mkv",
            "webm",
            "mov",
            "avi",
            "m4v",
            "mp3",
            "wav",
            "m4a",
            "aac",
            "flac",
            "ogg",
            "opus",
          ],
        },
      ],
    });
    if (selected === null) return;
    const path = typeof selected === "string" ? selected : selected[0];
    if (path) {
      setLocalPath(path);
      setUrl("");
    }
  }, [t]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const f = e.dataTransfer.files?.[0];
    if (f && "path" in f && typeof (f as File & { path?: string }).path === "string") {
      setLocalPath((f as File & { path: string }).path);
      setUrl("");
    }
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  return (
    <div className="flex flex-col gap-6 p-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-semibold text-text mb-2">
          {t("mediaTranscribe.title")}
        </h1>
        <p className="text-mid-gray text-sm">{t("mediaTranscribe.subtitle")}</p>
      </div>

      <div
        className="border border-dashed border-mid-gray/40 rounded-xl p-10 flex flex-col items-center justify-center gap-3 bg-mid-gray/5 min-h-[200px]"
        onDrop={onDrop}
        onDragOver={onDragOver}
      >
        <Film className="w-12 h-12 text-mid-gray/60" aria-hidden />
        <p className="text-lg font-medium text-text text-center">
          {t("mediaTranscribe.dropHint")}
        </p>
        <p className="text-sm text-mid-gray text-center max-w-md">
          {t("mediaTranscribe.dropSubhint")}
        </p>
      </div>

      <div className="flex flex-col sm:flex-row sm:flex-wrap sm:items-center gap-3">
        <Button type="button" onClick={onPickFile} disabled={busy}>
          {t("mediaTranscribe.selectFile")}
        </Button>
        <span className="text-mid-gray text-sm self-center">
          {t("mediaTranscribe.or")}
        </span>
        <div className="flex flex-1 min-w-0 items-center gap-2 flex-col sm:flex-row">
          <div className="relative flex-1 min-w-0 w-full">
            <Link2
              className="absolute start-3 top-1/2 -translate-y-1/2 w-4 h-4 text-mid-gray"
              aria-hidden
            />
            <Input
              className="ps-9"
              type="url"
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                if (e.target.value.trim()) setLocalPath(null);
              }}
              placeholder={t("mediaTranscribe.urlPlaceholder")}
              disabled={busy}
            />
          </div>
          <Button
            type="button"
            variant="secondary"
            onClick={runTranscription}
            disabled={!canTranscribe}
            className="shrink-0 w-full sm:w-auto"
          >
            {busy ? t("mediaTranscribe.loading") : t("mediaTranscribe.transcribe")}
          </Button>
        </div>
      </div>

      <label className="flex items-center gap-2 cursor-pointer text-sm text-text">
        <input
          type="checkbox"
          checked={applyVad}
          onChange={(e) => setApplyVad(e.target.checked)}
          disabled={busy}
          className="rounded border-mid-gray/40"
        />
        {t("mediaTranscribe.applyVad")}
      </label>

      {localPath ? (
        <p className="text-xs text-mid-gray break-all">
          {t("mediaTranscribe.selectedFile")}: {localPath}
        </p>
      ) : null}

      {resultText ? (
        <div className="flex flex-col gap-2">
          <h2 className="text-sm font-medium text-text">
            {t("mediaTranscribe.resultHeading")}
          </h2>
          <textarea
            readOnly
            className="w-full min-h-[160px] rounded-lg border border-mid-gray/20 bg-background p-3 text-sm text-text font-mono"
            value={resultText}
          />
        </div>
      ) : null}
    </div>
  );
};
