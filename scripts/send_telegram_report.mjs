import { readFile } from "node:fs/promises";

const DEFAULT_TIMEOUT_MS = 10000;
const TELEGRAM_MAX_MESSAGE_LENGTH = 4096;
const TELEGRAM_CHUNK_BODY_LENGTH = 4000;
const DEFAULT_REPORT_PATH = "artifacts/telegram_report.md";

export function getTelegramApiUrl(botToken, method) {
  return `https://api.telegram.org/bot${botToken}/${method}`;
}

export async function postTelegram(
  botToken,
  method,
  payload,
  options = {},
) {
  const url = getTelegramApiUrl(botToken, method);
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  );

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const json = await response.json().catch(() => null);
    if (!response.ok || !json?.ok) {
      throw new Error(
        json?.description ?? `Telegram ${method} failed: HTTP ${response.status}`,
      );
    }

    return json.result;
  } finally {
    clearTimeout(timeout);
  }
}

export function chunkTelegramMessage(text) {
  if (text.length <= TELEGRAM_MAX_MESSAGE_LENGTH) {
    return [text];
  }

  const chunks = [];
  let remaining = text;
  while (remaining.length > TELEGRAM_CHUNK_BODY_LENGTH) {
    let splitAt = remaining.lastIndexOf("\n\n", TELEGRAM_CHUNK_BODY_LENGTH);
    if (splitAt < TELEGRAM_CHUNK_BODY_LENGTH * 0.5) {
      splitAt = remaining.lastIndexOf("\n", TELEGRAM_CHUNK_BODY_LENGTH);
    }
    if (splitAt < TELEGRAM_CHUNK_BODY_LENGTH * 0.5) {
      splitAt = TELEGRAM_CHUNK_BODY_LENGTH;
    }
    chunks.push(remaining.slice(0, splitAt).trim());
    remaining = remaining.slice(splitAt).trimStart();
  }
  if (remaining) {
    chunks.push(remaining);
  }
  return chunks;
}

async function main() {
  const botToken = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;
  const reportPath = process.argv[2] ?? DEFAULT_REPORT_PATH;

  if (!botToken) {
    throw new Error("Missing TELEGRAM_BOT_TOKEN.");
  }
  if (!chatId) {
    throw new Error("Missing TELEGRAM_CHAT_ID.");
  }

  const report = await readFile(reportPath, "utf8");
  const chunks = chunkTelegramMessage(report);
  for (const [index, chunk] of chunks.entries()) {
    const suffix = chunks.length > 1 ? `\n\nPart ${index + 1}/${chunks.length}` : "";
    await postTelegram(botToken, "sendMessage", {
      chat_id: chatId,
      text: `${chunk}${suffix}`,
      disable_web_page_preview: true,
    });
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
