import { getContext } from '../../../extensions.js';

const EXTENSION_KEY = 'quiet-stream-keepalive';
const GENERATE_PATH = '/api/backends/chat-completions/generate';
const SETTINGS_DOM_ID = 'qsk_settings_container';

const DEFAULT_SETTINGS = {
    enabled: true,
    onlyQuiet: false,
    showProgressToast: true,
    toastDelayMs: 5000,
};

/** @type {typeof window.fetch | null} */
let originalFetch = null;
let installed = false;
let activeCount = 0;

function getSettings() {
    const ctx = getContext();
    if (!ctx.extensionSettings[EXTENSION_KEY]) {
        ctx.extensionSettings[EXTENSION_KEY] = { ...DEFAULT_SETTINGS };
    }
    const s = ctx.extensionSettings[EXTENSION_KEY];
    for (const [k, v] of Object.entries(DEFAULT_SETTINGS)) {
        if (typeof s[k] === 'undefined') {
            s[k] = v;
        }
    }
    return s;
}

function saveSettings() {
    getContext().saveSettingsDebounced?.();
}

function safeJsonParse(text, fallback = null) {
    try {
        return JSON.parse(text);
    } catch {
        return fallback;
    }
}

/**
 * 判断是否需要拦截并转换为“流式发出 + 前端拼回 JSON”
 */
function shouldIntercept(input, init) {
    const settings = getSettings();
    if (!settings.enabled) return null;

    const urlStr = input instanceof Request ? input.url : String(input);
    let pathname = '';
    try {
        pathname = new URL(urlStr, window.location.origin).pathname;
    } catch {
        return null;
    }
    if (!pathname.endsWith(GENERATE_PATH)) return null;

    const method = (init?.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();
    if (method !== 'POST') return null;

    if (typeof init?.body !== 'string') return null;
    const payload = safeJsonParse(init.body, null);
    if (!payload || typeof payload !== 'object') return null;

    // 原本就是 stream: true 的请求不干预
    if (payload.stream !== false) return null;

    if (settings.onlyQuiet && payload.type !== 'quiet') {
        return null;
    }

    return payload;
}

/**
 * 创建多渠道兼容的 SSE -> 非流式 JSON 拼装器
 */
function createStreamAssembler(sourceHint) {
    let detectedFormat = detectInitialFormat(sourceHint);
    let chunkCount = 0;
    let errorObject = null;

    // 1. OpenAI / Custom / OpenRouter / DeepSeek / xAI / Mistral 等通用状态
    const oai = {
        id: '',
        model: '',
        created: Math.floor(Date.now() / 1000),
        system_fingerprint: undefined,
        usage: undefined,
        /** @type {Map<number, any>} */
        choices: new Map(),
    };

    // 2. Anthropic Claude 状态
    const claude = {
        id: '',
        model: '',
        role: 'assistant',
        stop_reason: null,
        stop_sequence: null,
        usage: undefined,
        /** @type {Map<number, any>} */
        blocks: new Map(),
    };

    // 3. Google MakerSuite / VertexAI 状态
    const gemini = {
        parts: [],
        usageMetadata: undefined,
        finishReason: undefined,
    };

    // 4. Cohere v2 状态
    const cohere = {
        id: '',
        text: '',
        toolPlan: '',
        /** @type {Map<number, any>} */
        toolCalls: new Map(),
        finish_reason: 'COMPLETE',
        usage: undefined,
    };

    function detectInitialFormat(source) {
        if (source === 'claude') return 'claude';
        if (source === 'makersuite' || source === 'vertexai') return 'gemini';
        if (source === 'cohere') return 'cohere';
        return 'openai';
    }

    function ensureOaiChoice(index) {
        if (!oai.choices.has(index)) {
            oai.choices.set(index, {
                index,
                message: {
                    role: 'assistant',
                    content: '',
                },
                _reasoning: '',
                _reasoningField: '',
                _reasoningDetails: [],
                _images: [],
                _toolCalls: new Map(),
                finish_reason: null,
                logprobs: undefined,
            });
        }
        return oai.choices.get(index);
    }

    function mergeToolCallDeltas(targetMap, deltas) {
        if (!Array.isArray(deltas)) return;
        for (let i = 0; i < deltas.length; i++) {
            const d = deltas[i];
            if (!d || typeof d !== 'object') continue;
            const idx = typeof d.index === 'number' ? d.index : i;
            if (!targetMap.has(idx)) {
                targetMap.set(idx, {
                    id: d.id || '',
                    type: d.type || 'function',
                    function: {
                        name: d.function?.name || '',
                        arguments: d.function?.arguments || '',
                    },
                    ...(d.extra_content ? { extra_content: d.extra_content } : {}),
                    ...(d.signature ? { signature: d.signature } : {}),
                });
                continue;
            }
            const cur = targetMap.get(idx);
            if (d.id) cur.id = d.id;
            if (d.type) cur.type = d.type;
            if (d.extra_content) cur.extra_content = d.extra_content;
            if (d.signature) cur.signature = d.signature;
            if (d.function) {
                if (d.function.name) cur.function.name += d.function.name;
                if (typeof d.function.arguments === 'string') {
                    cur.function.arguments += d.function.arguments;
                }
            }
        }
    }

    function pushGeminiTextPart(text, isThought, signature) {
        if (!text && !signature) return;
        const last = gemini.parts[gemini.parts.length - 1];
        if (
            last &&
            typeof last.text === 'string' &&
            Boolean(last.thought) === Boolean(isThought) &&
            !last.functionCall &&
            !last.inlineData &&
            !last.thoughtSignature
        ) {
            last.text += text || '';
            if (signature) last.thoughtSignature = signature;
            return;
        }
        const part = { text: text || '' };
        if (isThought) part.thought = true;
        if (signature) part.thoughtSignature = signature;
        gemini.parts.push(part);
    }

    function processEvent(eventName, dataText) {
        const trimmed = dataText.trim();
        if (!trimmed) return { done: false };
        if (trimmed === '[DONE]') return { done: true };

        const data = safeJsonParse(trimmed, null);
        if (!data || typeof data !== 'object') return { done: false };

        chunkCount++;

        // 错误包检测（任意渠道）
        if (data.error || eventName === 'error' || data.type === 'error') {
            errorObject = data.error ? { error: data.error } : { error: data };
            return { done: false };
        }

        // 动态识别真实响应格式（兼容反代混用场景）
        if (Array.isArray(data.candidates)) {
            detectedFormat = 'gemini';
        } else if (
            ['message_start', 'content_block_start', 'content_block_delta', 'content_block_stop', 'message_delta', 'message_stop'].includes(data.type)
        ) {
            detectedFormat = 'claude';
        } else if (
            ['message-start', 'content-start', 'content-delta', 'content-end', 'tool-plan-start', 'tool-plan-delta', 'tool-plan-end', 'tool-call-start', 'tool-call-delta', 'tool-call-end', 'message-end'].includes(data.type)
        ) {
            detectedFormat = 'cohere';
        } else if (Array.isArray(data.choices)) {
            detectedFormat = 'openai';
        }

        if (detectedFormat === 'gemini') {
            if (data.usageMetadata) gemini.usageMetadata = data.usageMetadata;
            const cand = data.candidates?.[0];
            if (cand?.finishReason) gemini.finishReason = cand.finishReason;
            const parts = cand?.content?.parts;
            if (Array.isArray(parts)) {
                for (const p of parts) {
                    if (!p || typeof p !== 'object') continue;
                    if (p.functionCall || p.inlineData) {
                        gemini.parts.push(p);
                    } else if (typeof p.text === 'string' || p.thoughtSignature) {
                        pushGeminiTextPart(p.text || '', Boolean(p.thought), p.thoughtSignature);
                    }
                }
            }
            return { done: false };
        }

        if (detectedFormat === 'claude') {
            switch (data.type) {
                case 'message_start':
                    if (data.message?.id) claude.id = data.message.id;
                    if (data.message?.model) claude.model = data.message.model;
                    if (data.message?.role) claude.role = data.message.role;
                    if (data.message?.usage) claude.usage = data.message.usage;
                    break;
                case 'content_block_start': {
                    const idx = typeof data.index === 'number' ? data.index : claude.blocks.size;
                    const blk = data.content_block ? { ...data.content_block } : { type: 'text', text: '' };
                    if (blk.type === 'tool_use') {
                        blk._partialJson = typeof blk.input === 'string' ? blk.input : '';
                    }
                    claude.blocks.set(idx, blk);
                    break;
                }
                case 'content_block_delta': {
                    const idx = typeof data.index === 'number' ? data.index : 0;
                    let blk = claude.blocks.get(idx);
                    const d = data.delta || {};
                    if (!blk) {
                        blk = d.type === 'thinking_delta'
                            ? { type: 'thinking', thinking: '' }
                            : d.type === 'input_json_delta'
                                ? { type: 'tool_use', id: '', name: '', _partialJson: '' }
                                : { type: 'text', text: '' };
                        claude.blocks.set(idx, blk);
                    }
                    if (d.type === 'text_delta' && typeof d.text === 'string') {
                        blk.text = (blk.text || '') + d.text;
                    } else if (d.type === 'thinking_delta' && typeof d.thinking === 'string') {
                        blk.thinking = (blk.thinking || '') + d.thinking;
                    } else if (d.type === 'signature_delta' && typeof d.signature === 'string') {
                        blk.signature = (blk.signature || '') + d.signature;
                    } else if (d.type === 'input_json_delta' && typeof d.partial_json === 'string') {
                        blk._partialJson = (blk._partialJson || '') + d.partial_json;
                    }
                    break;
                }
                case 'message_delta':
                    if (data.delta?.stop_reason) claude.stop_reason = data.delta.stop_reason;
                    if (data.delta?.stop_sequence !== undefined) claude.stop_sequence = data.delta.stop_sequence;
                    if (data.usage) claude.usage = { ...(claude.usage || {}), ...data.usage };
                    break;
                case 'message_stop':
                    return { done: true };
            }
            return { done: false };
        }

        if (detectedFormat === 'cohere') {
            switch (data.type) {
                case 'message-start':
                    if (data.id) cohere.id = data.id;
                    break;
                case 'content-delta':
                    cohere.text += data.delta?.message?.content?.text || '';
                    break;
                case 'tool-plan-delta':
                    cohere.toolPlan += data.delta?.message?.tool_plan || '';
                    break;
                case 'tool-call-start': {
                    const idx = typeof data.index === 'number' ? data.index : cohere.toolCalls.size;
                    const tc = data.delta?.message?.tool_calls;
                    if (tc) {
                        cohere.toolCalls.set(idx, {
                            id: tc.id || '',
                            type: tc.type || 'function',
                            function: {
                                name: tc.function?.name || '',
                                arguments: tc.function?.arguments || '',
                            },
                        });
                    }
                    break;
                }
                case 'tool-call-delta': {
                    const idx = typeof data.index === 'number' ? data.index : 0;
                    const cur = cohere.toolCalls.get(idx);
                    const args = data.delta?.message?.tool_calls?.function?.arguments;
                    if (cur && typeof args === 'string') {
                        cur.function.arguments += args;
                    }
                    break;
                }
                case 'message-end':
                    if (data.delta?.finish_reason) cohere.finish_reason = data.delta.finish_reason;
                    if (data.delta?.usage) cohere.usage = data.delta.usage;
                    return { done: true };
            }
            return { done: false };
        }

        // 默认 OpenAI / Custom / OpenRouter / DeepSeek / xAI / Mistral SSE
        if (data.id) oai.id = data.id;
        if (data.model) oai.model = data.model;
        if (data.created) oai.created = data.created;
        if (data.system_fingerprint) oai.system_fingerprint = data.system_fingerprint;
        if (data.usage) oai.usage = data.usage;

        if (Array.isArray(data.choices)) {
            for (const c of data.choices) {
                if (!c || typeof c !== 'object') continue;
                const idx = typeof c.index === 'number' ? c.index : 0;
                const entry = ensureOaiChoice(idx);

                const d = c.delta || {};
                const m = c.message || {};

                if (d.role || m.role) {
                    entry.message.role = d.role || m.role;
                }

                // 正文内容（兼容字符串与 Mistral 数组格式）
                const contentPiece = d.content ?? m.content ?? c.text;
                if (typeof contentPiece === 'string') {
                    entry.message.content += contentPiece;
                } else if (Array.isArray(contentPiece)) {
                    for (const part of contentPiece) {
                        if (typeof part?.text === 'string') {
                            entry.message.content += part.text;
                        }
                        if (Array.isArray(part?.thinking)) {
                            for (const th of part.thinking) {
                                if (typeof th?.text === 'string') {
                                    entry._reasoning += th.text;
                                    if (!entry._reasoningField) entry._reasoningField = 'reasoning_content';
                                }
                            }
                        }
                    }
                }

                // 思考内容（兼容 reasoning_content 与 OpenRouter reasoning）
                if (typeof d.reasoning_content === 'string') {
                    entry._reasoning += d.reasoning_content;
                    if (!entry._reasoningField) entry._reasoningField = 'reasoning_content';
                } else if (typeof d.reasoning === 'string') {
                    entry._reasoning += d.reasoning;
                    if (!entry._reasoningField) entry._reasoningField = 'reasoning';
                }
                if (typeof m.reasoning_content === 'string') {
                    entry._reasoning += m.reasoning_content;
                    if (!entry._reasoningField) entry._reasoningField = 'reasoning_content';
                } else if (typeof m.reasoning === 'string') {
                    entry._reasoning += m.reasoning;
                    if (!entry._reasoningField) entry._reasoningField = 'reasoning';
                }

                // OpenRouter reasoning_details / images
                if (Array.isArray(d.reasoning_details)) entry._reasoningDetails.push(...d.reasoning_details);
                if (Array.isArray(m.reasoning_details)) entry._reasoningDetails.push(...m.reasoning_details);
                if (Array.isArray(d.images)) entry._images.push(...d.images);

                // 工具调用
                mergeToolCallDeltas(entry._toolCalls, d.tool_calls);
                mergeToolCallDeltas(entry._toolCalls, m.tool_calls);

                if (c.finish_reason) entry.finish_reason = c.finish_reason;
                if (c.logprobs !== undefined) entry.logprobs = c.logprobs;
            }
        }

        return { done: false };
    }

    function getProgressStats() {
        let reasoningChars = 0;
        let contentChars = 0;
        if (detectedFormat === 'gemini') {
            for (const p of gemini.parts) {
                if (typeof p.text === 'string') {
                    if (p.thought) reasoningChars += p.text.length;
                    else contentChars += p.text.length;
                }
            }
        } else if (detectedFormat === 'claude') {
            for (const b of claude.blocks.values()) {
                if (b.type === 'thinking') reasoningChars += (b.thinking || '').length;
                if (b.type === 'text') contentChars += (b.text || '').length;
            }
        } else if (detectedFormat === 'cohere') {
            contentChars = cohere.text.length;
        } else {
            const first = oai.choices.get(0);
            if (first) {
                reasoningChars = first._reasoning.length;
                contentChars = first.message.content.length;
            }
        }
        return { chunkCount, reasoningChars, contentChars };
    }

    function buildFinalJson() {
        if (errorObject) return errorObject;
        if (chunkCount === 0) {
            return { error: { message: 'SSE stream ended with no data chunks.' } };
        }

        if (detectedFormat === 'gemini') {
            const responseText = gemini.parts
                .filter(p => !p.thought && typeof p.text === 'string')
                .map(p => p.text)
                .join('\n\n');
            return {
                choices: [{ index: 0, message: { role: 'assistant', content: responseText }, finish_reason: gemini.finishReason || 'STOP' }],
                responseContent: { role: 'model', parts: gemini.parts },
                ...(gemini.usageMetadata ? { usageMetadata: gemini.usageMetadata } : {}),
            };
        }

        if (detectedFormat === 'claude') {
            const content = [...claude.blocks.entries()]
                .sort((a, b) => a[0] - b[0])
                .map(([, blk]) => {
                    const out = { ...blk };
                    if (out.type === 'tool_use' && typeof out._partialJson === 'string') {
                        out.input = out._partialJson ? safeJsonParse(out._partialJson, {}) : (out.input || {});
                    }
                    delete out._partialJson;
                    return out;
                });
            return {
                id: claude.id || `msg_${Date.now()}`,
                type: 'message',
                role: claude.role || 'assistant',
                model: claude.model || '',
                content,
                stop_reason: claude.stop_reason || 'end_turn',
                stop_sequence: claude.stop_sequence,
                ...(claude.usage ? { usage: claude.usage } : {}),
            };
        }

        if (detectedFormat === 'cohere') {
            const toolCalls = [...cohere.toolCalls.entries()]
                .sort((a, b) => a[0] - b[0])
                .map(([, tc]) => tc);
            return {
                id: cohere.id || `cohere_${Date.now()}`,
                finish_reason: cohere.finish_reason,
                message: {
                    role: 'assistant',
                    content: cohere.text ? [{ type: 'text', text: cohere.text }] : [],
                    ...(cohere.toolPlan ? { tool_plan: cohere.toolPlan } : {}),
                    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
                },
                ...(cohere.usage ? { usage: cohere.usage } : {}),
            };
        }

        // OpenAI 通用拼装
        const choices = [...oai.choices.entries()]
            .sort((a, b) => a[0] - b[0])
            .map(([index, entry]) => {
                const message = {
                    role: entry.message.role || 'assistant',
                    content: entry.message.content,
                };
                if (entry._reasoning) {
                    message.reasoning_content = entry._reasoning;
                    if (sourceHint === 'openrouter' || entry._reasoningField === 'reasoning') {
                        message.reasoning = entry._reasoning;
                    }
                }
                if (entry._reasoningDetails.length > 0) {
                    message.reasoning_details = entry._reasoningDetails;
                }
                if (entry._images.length > 0) {
                    message.images = entry._images;
                }
                if (entry._toolCalls.size > 0) {
                    message.tool_calls = [...entry._toolCalls.entries()]
                        .sort((a, b) => a[0] - b[0])
                        .map(([, tc]) => tc);
                }
                return {
                    index,
                    message,
                    finish_reason: entry.finish_reason || (message.tool_calls ? 'tool_calls' : 'stop'),
                    ...(entry.logprobs !== undefined ? { logprobs: entry.logprobs } : {}),
                };
            });

        return {
            id: oai.id || `chatcmpl_${Date.now()}`,
            object: 'chat.completion',
            created: oai.created,
            model: oai.model || '',
            ...(oai.system_fingerprint ? { system_fingerprint: oai.system_fingerprint } : {}),
            choices,
            ...(oai.usage ? { usage: oai.usage } : {}),
        };
    }

    return { processEvent, getProgressStats, buildFinalJson };
}

/**
 * 读取 SSE ReadableStream 并拼装回普通 JSON Response
 */
async function consumeSseToJsonResponse(response, payload, signal) {
    const settings = getSettings();
    const assembler = createStreamAssembler(payload.chat_completion_source);
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    const startedAt = Date.now();

    let buffer = '';
    let toastEl = null;
    let progressTimer = null;

    const updateToast = () => {
        if (!settings.showProgressToast || !globalThis.toastr) return;
        const elapsed = Math.floor((Date.now() - startedAt) / 1000);
        const { reasoningChars, contentChars } = assembler.getProgressStats();
        const label = payload.type === 'quiet' ? '后台变量/静默生成中' : '非流式保活接收中';
        const detail = `${label}（已耗时 ${elapsed}s｜思考 ${reasoningChars} 字｜正文 ${contentChars} 字）`;

        if (!toastEl) {
            toastEl = globalThis.toastr.info(detail, 'Stream Keepalive', {
                timeOut: 0,
                extendedTimeOut: 0,
                tapToDismiss: false,
            });
        } else {
            toastEl.find('.toast-message').text(detail);
        }
    };

    if (settings.showProgressToast) {
        const delayMs = Math.max(1000, Number(settings.toastDelayMs) || 5000);
        progressTimer = window.setInterval(() => {
            if (Date.now() - startedAt >= delayMs) {
                updateToast();
            }
        }, 1000);
    }

    const onAbort = () => {
        try {
            reader.cancel();
        } catch {
            // ignore
        }
    };
    signal?.addEventListener?.('abort', onAbort, { once: true });

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            buffer = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

            const frames = buffer.split('\n\n');
            buffer = frames.pop() ?? '';

            for (const frame of frames) {
                const result = parseSseFrame(frame, assembler);
                if (result.done) {
                    try { await reader.cancel(); } catch { /* ignore */ }
                    const finalJson = assembler.buildFinalJson();
                    return new Response(JSON.stringify(finalJson), {
                        status: 200,
                        headers: { 'Content-Type': 'application/json; charset=utf-8' },
                    });
                }
            }
        }

        // 处理末尾残留（部分上游不会发 [DONE] 或末尾缺少空行）
        buffer += decoder.decode();
        if (buffer.trim()) {
            parseSseFrame(buffer, assembler);
        }

        const finalJson = assembler.buildFinalJson();
        return new Response(JSON.stringify(finalJson), {
            status: 200,
            headers: { 'Content-Type': 'application/json; charset=utf-8' },
        });
    } finally {
        signal?.removeEventListener?.('abort', onAbort);
        if (progressTimer !== null) window.clearInterval(progressTimer);
        if (toastEl && globalThis.toastr) {
            globalThis.toastr.clear(toastEl);
        }
    }
}

function parseSseFrame(frame, assembler) {
    let eventName = 'message';
    const dataLines = [];

    for (const line of frame.split('\n')) {
        if (!line || line.startsWith(':')) continue; // 跳过空行与 SSE 注释保活行
        const colon = line.indexOf(':');
        const field = colon === -1 ? line : line.slice(0, colon);
        let value = colon === -1 ? '' : line.slice(colon + 1);
        if (value.startsWith(' ')) value = value.slice(1);

        if (field === 'event') {
            eventName = value;
        } else if (field === 'data') {
            dataLines.push(value);
        }
    }

    if (dataLines.length === 0) return { done: false };
    return assembler.processEvent(eventName, dataLines.join('\n'));
}

function installFetchWrapper() {
    if (installed) return;
    originalFetch = window.fetch.bind(window);
    installed = true;

    window.fetch = async function wrappedFetch(input, init) {
        const payload = shouldIntercept(input, init);
        if (!payload) {
            return originalFetch(input, init);
        }

        const nextPayload = { ...payload, stream: true };
        const nextInit = {
            ...init,
            body: JSON.stringify(nextPayload),
        };

        activeCount++;
        try {
            const response = await originalFetch(input, nextInit);

            // 非 2xx 或上游未返回 SSE（例如直接返回了错误 JSON），原样交回给酒馆处理
            const contentType = (response.headers.get('content-type') || '').toLowerCase();
            if (!response.ok || !response.body || !contentType.includes('text/event-stream')) {
                return response;
            }

            return await consumeSseToJsonResponse(response, nextPayload, nextInit.signal);
        } finally {
            activeCount = Math.max(0, activeCount - 1);
        }
    };
}

function uninstallFetchWrapper() {
    if (!installed || !originalFetch) return;
    window.fetch = originalFetch;
    installed = false;
}

function renderSettingsUI() {
    if (document.getElementById(SETTINGS_DOM_ID)) return;
    const host = document.getElementById('extensions_settings2') || document.getElementById('extensions_settings');
    if (!host) return;

    const settings = getSettings();
    const container = document.createElement('div');
    container.id = SETTINGS_DOM_ID;
    container.className = 'extension_container';
    container.innerHTML = `
        <div class="inline-drawer">
            <div class="inline-drawer-toggle inline-toggle-header">
                <b>Quiet Stream Keepalive（后台非流式保活）</b>
                <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
            </div>
            <div class="inline-drawer-content">
                <label class="checkbox_label">
                    <input type="checkbox" id="qsk_enabled" />
                    <span>启用流式保活代理（将非流式请求改为流式接收后再拼回 JSON）</span>
                </label>
                <label class="checkbox_label">
                    <input type="checkbox" id="qsk_only_quiet" />
                    <span>仅拦截 quiet 后台请求（取消勾选则拦截所有非流式请求）</span>
                </label>
                <label class="checkbox_label">
                    <input type="checkbox" id="qsk_show_toast" />
                    <span>等待超过 5 秒后在右上角显示实时思考/接收字数</span>
                </label>
            </div>
        </div>
    `;

    host.appendChild(container);

    const enabledEl = /** @type {HTMLInputElement} */ (document.getElementById('qsk_enabled'));
    const onlyQuietEl = /** @type {HTMLInputElement} */ (document.getElementById('qsk_only_quiet'));
    const showToastEl = /** @type {HTMLInputElement} */ (document.getElementById('qsk_show_toast'));

    enabledEl.checked = Boolean(settings.enabled);
    onlyQuietEl.checked = Boolean(settings.onlyQuiet);
    showToastEl.checked = Boolean(settings.showProgressToast);

    enabledEl.addEventListener('change', () => {
        settings.enabled = enabledEl.checked;
        saveSettings();
    });
    onlyQuietEl.addEventListener('change', () => {
        settings.onlyQuiet = onlyQuietEl.checked;
        saveSettings();
    });
    showToastEl.addEventListener('change', () => {
        settings.showProgressToast = showToastEl.checked;
        saveSettings();
    });
}

export function init() {
    getSettings();
    installFetchWrapper();
    renderSettingsUI();
    console.log('[quiet-stream-keepalive] 已启用（全渠道通用 SSE 保活拼装）');
}

export function onEnable() {
    installFetchWrapper();
    renderSettingsUI();
}

export function onDisable() {
    uninstallFetchWrapper();
    document.getElementById(SETTINGS_DOM_ID)?.remove();
}
