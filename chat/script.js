// Data stores
const codeBlocks = new Map();
const attachedFileObjects = new Map();
let codeBlockIdCounter = 0;
let attachedFileIdCounter = 0;

// DOM Elements
const appContainer = document.querySelector('.app-container');
const settingsForm = document.getElementById('settings-form');
const settingsToggleBtn = document.getElementById('settings-toggle');
const ollamaUrlInput = document.getElementById('ollama-url-input');
const modelInput = document.getElementById('model-input');
const chatContainer = document.getElementById('chat-container');
const chatHistory = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form');
const inputWrapper = document.getElementById('input-wrapper');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const errorMessageDiv = document.getElementById('error-message');
const fileInput = document.getElementById('file-input');
const fileInputLabel = document.getElementById('file-input-label');
const filePreview = document.getElementById('file-preview');
const toastNotification = document.getElementById('toast-notification');
const saveHistoryBtn = document.getElementById('save-history-btn');
const restoreHistoryInput = document.getElementById('restore-history-input');
const appTitle = document.getElementById('app-title');

// State
let conversationHistory = [];
let messageTimestamps = [];
let attachedFiles = [];
let loadingIndicator;
let debugMode = null;
let audioContext = null;

function updateAppTitle() {
    const modelName = modelInput.value.trim();
    appTitle.textContent = modelName ? `Ollama Chat (${modelName})` : 'Ollama Chat';
}

function displayInitialWelcomeMessage() {
    const welcomeText = "ページの一番下にある枠にメッセージを入力し、送信するとAIが回答します。このページはローカルのOllamaサーバーと通信します。";
    const timestamp = new Date().toISOString();
    if (messageTimestamps.length === 0) {
        messageTimestamps.push(timestamp);
         // UI表示用の履歴と、API送信用（内部状態）の履歴を分ける
        conversationHistory.push({role: 'model', parts: [{ text: welcomeText }], timestamp: timestamp });
    }
}

function renderFormattedContent(container, text, timestamp) {
    const placeholders = new Map();
    let placeholderId = 0;

    const codeBlockRegex = /```(\w*)\n([\s\S]*?)\n```/g;
    let processedText = text.replace(codeBlockRegex, (match, lang, code) => {
        const id = `%%CODE_BLOCK_PLACEHOLDER_${placeholderId++}%%`;
        const language = (lang || 'plaintext').toLowerCase();

        const isSvg = (language === 'svg' || language === 'xml') && code.trim().startsWith('<svg');
        let blockHtml;
        if (isSvg) {
            blockHtml = `<div class="svg-container">${code}</div>`;
        } else {
            const blockId = `code-block-${codeBlockIdCounter++}`;
            const highlightedCode = hljs.getLanguage(language)
                ? hljs.highlight(code, { language, ignoreIllegals: true }).value
                : hljs.highlightAuto(code).value;

            codeBlocks.set(blockId, { code, language, timestamp });

            blockHtml = `
                <div class="code-block-wrapper" data-block-id="${blockId}">
                    <div class="code-header">
                        <span class="language-name">${language === 'plaintext' ? 'text' : language}</span>
                        <div class="code-actions">
                            <button class="code-copy-btn">Copy</button>
                            <button class="code-download-btn">Download</button>
                        </div>
                    </div>
                    <pre><code class="hljs language-${language}">${highlightedCode}</code></pre>
                </div>
            `;
        }

        placeholders.set(id, blockHtml);
        return id;
    });

    let html = '';
    if (typeof marked !== 'undefined') {
        html = marked.parse(processedText);
    } else {
        html = processedText.replace(/\n/g, '<br>');
    }

    html = html.replace(/<p>(%%CODE_BLOCK_PLACEHOLDER_\d+%%)<\/p>/g, (match, placeholder) => {
        return placeholders.get(placeholder) || '';
    });
    html = html.replace(/(%%CODE_BLOCK_PLACEHOLDER_\d+%%)/g, (match, placeholder) => {
        return placeholders.get(placeholder) || '';
    });

    container.innerHTML = html;

    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
        Promise.resolve().then(() => MathJax.typesetPromise([container]))
            .catch((err) => console.error('MathJax typesetting error:', err))
            .finally(() => chatHistory.scrollTop = chatHistory.scrollHeight);
    }
}

function addMessageToDisplay(sender, text, files = [], timestamp = null) {
    const messageDate = timestamp ? new Date(timestamp) : new Date();
    const messageRow = document.createElement('div');
    messageRow.classList.add('message-row', sender === 'user' ? 'user' : 'ai');

    const messageEl = document.createElement('div');
    messageEl.classList.add('message', sender === 'user' ? 'user-message' : 'ai-message');

    const contentEl = document.createElement('div');
    contentEl.classList.add('message-content');

    if (text) {
        renderFormattedContent(contentEl, text, timestamp);
    }

    if (sender === 'user' && files.length > 0) {
        const filesContainer = document.createElement('div');
        filesContainer.className = 'message-files-container';
        if (text) filesContainer.style.marginTop = '8px';

        files.forEach(file => {
            const isFileObject = file instanceof File;
            const isReconstructed = !!file.isReconstructed;
            const mimeType = isFileObject ? file.type : (isReconstructed ? file.inlineData.mimeType : '');
            const fileName = isFileObject ? file.name : (isReconstructed ? file.name : `添付ファイル`);
            const isImage = mimeType && mimeType.startsWith('image/');

            let fileElement;
            let fileId = null;

            if (isReconstructed) {
                const byteString = atob(file.inlineData.data);
                const ab = new ArrayBuffer(byteString.length);
                const ia = new Uint8Array(ab);
                for (let i = 0; i < byteString.length; i++) {
                    ia[i] = byteString.charCodeAt(i);
                }
                const blob = new Blob([ab], { type: mimeType });
                const reconstructedFile = new File([blob], fileName, { type: mimeType });

                fileId = `file-${attachedFileIdCounter++}`;
                attachedFileObjects.set(fileId, reconstructedFile);

            } else if (isFileObject) {
                fileId = `file-${attachedFileIdCounter++}`;
                attachedFileObjects.set(fileId, file);
            }

            if (isImage) {
                const img = document.createElement('img');
                if (fileId) {
                   const fileObj = attachedFileObjects.get(fileId);
                   img.src = URL.createObjectURL(fileObj);
                   img.onload = () => URL.revokeObjectURL(img.src);
                } else {
                   img.src = `data:${mimeType};base64,${file.inlineData.data}`;
                }
                fileElement = img;
            } else {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-display-item';
                fileItem.innerHTML = `<svg viewBox="0 0 24 24"><path fill="currentColor" d="M16.5,6V17.5A4,4 0 0,1 12.5,21.5A4,4 0 0,1 8.5,17.5V5A2.5,2.5 0 0,1 11,2.5A2.5,2.5 0 0,1 13.5,5V15.5A1,1 0 0,1 12.5,16.5A1,1 0 0,1 11.5,15.5V6H10V15.5A2.5,2.5 0 0,0 12.5,18A2.5,2.5 0 0,0 15,15.5V5A4,4 0 0,0 11,1A4,4 0 0,0 7,5V17.5A5.5,5.5 0 0,0 12.5,23A5.5,5.5 0 0,0 18,17.5V6H16.5Z" /></svg><span>${fileName}</span>`;
                fileElement = fileItem;
            }

            if (fileId) {
                fileElement.dataset.fileId = fileId;
                fileElement.title = `クリックしてダウンロード: ${fileName}`;
            }

            filesContainer.appendChild(fileElement);
        });
        contentEl.appendChild(filesContainer);
    }

    messageEl.appendChild(contentEl);

    const timestampEl = document.createElement('div');
    timestampEl.classList.add('message-timestamp');
    timestampEl.textContent = messageDate.toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' });

    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-message-btn';
    copyBtn.title = 'メッセージをコピー';
    copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>`;
    copyBtn.setAttribute('data-copy-text', text);

    const metaContainer = document.createElement('div');
    metaContainer.className = 'meta-container';
    metaContainer.appendChild(timestampEl);
    metaContainer.appendChild(copyBtn);

    messageRow.appendChild(messageEl);
    messageRow.appendChild(metaContainer);

    chatHistory.appendChild(messageRow);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function playNotificationSound() {
    if (!audioContext || audioContext.state !== 'running') {
        console.warn("AudioContext not available, skipping sound playback.");
        return;
    }
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    const now = audioContext.currentTime;
    const duration = 0.15;
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(880, now);
    gainNode.gain.setValueAtTime(0, now);
    gainNode.gain.linearRampToValueAtTime(0.3, now + 0.01);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, now + duration);
    oscillator.start(now);
    oscillator.stop(now + duration);
}

// ヘルパー関数: ファイルがテキスト形式かを判定
function isTextFile(file) {
    const textMimeTypes = [
        'text/',
        'application/json',
        'application/xml',
        'application/javascript',
        'application/x-javascript',
        'application/sql'
    ];
    // MIMEタイプが不明な場合やOSによって異なる場合を考慮し、拡張子でも補完的に判定
    const textExtensions = ['.txt', '.js', '.ts', '.py', '.java', '.c', '.cpp', '.cs', '.html', '.css', '.scss', '.json', '.xml', '.md', '.sql', '.sh', '.rb', '.go', '.php', '.yaml', '.yml', '.ini', '.cfg', '.log'];
    const fileName = file.name.toLowerCase();

    // MIMEタイプで判定
    if (textMimeTypes.some(type => file.type.startsWith(type))) {
        return true;
    }
    // MIMEタイプが空の場合、拡張子で判定
    if (file.type === '' && textExtensions.some(ext => fileName.endsWith(ext))) {
        return true;
    }
    return false;
}

async function handleChatSubmit(e) {
    e.preventDefault();
    const userInput = messageInput.value.trim();
    if (!userInput && attachedFiles.length === 0) return;

    if (debugMode) {
        // (Omitted for brevity)
        return;
    }

    inputWrapper.classList.add('submitting');
    sendButton.disabled = true; messageInput.disabled = true; fileInput.disabled = true;
    fileInputLabel.style.cssText = 'cursor: not-allowed; opacity: 0.5;';
    document.querySelectorAll('.remove-file').forEach(btn => btn.style.display = 'none');

    const filesToSend = [...attachedFiles];
    const userTimestamp = new Date().toISOString();

    // UIには元のメッセージとファイルアイコンを表示
    addMessageToDisplay('user', userInput, filesToSend, userTimestamp);

    loadingIndicator.style.display = 'block';
    chatHistory.appendChild(loadingIndicator);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    try {
        // ユーザープロンプトと画像パーツを準備
        let processedUserInput = userInput;
        const imagePartsForHistory = [];

        // 添付ファイルをテキストと画像に分けて処理
        for (const file of filesToSend) {
            if (isTextFile(file)) {
                // テキストファイルの場合: 内容を読み込みプロンプトに追加
                const fileContent = await new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result);
                    reader.onerror = reject;
                    reader.readAsText(file);
                });
                // AIが認識しやすいように区切りマーカーを追加
                processedUserInput += `\n\n--- START OF FILE: ${file.name} ---\n${fileContent}\n--- END OF FILE: ${file.name} ---`;
            } else {
                // 画像ファイルの場合: Base64エンコードしてパーツに追加
                const part = await fileToGenerativePartForHistory(file);
                imagePartsForHistory.push(part);
            }
        }

        // API送信用と履歴保存用のデータを作成
        const userPartsForHistory = [{ text: processedUserInput }, ...imagePartsForHistory];
        conversationHistory.push({ role: 'user', parts: userPartsForHistory, timestamp: userTimestamp });

        const messagesForOllama = conversationHistory.map(entry => {
            // ウェルカムメッセージはAPIに送信しない
            if (entry.role === 'model' && entry.timestamp === messageTimestamps[0]) {
                return null;
            }
            const role = entry.role === 'model' ? 'assistant' : 'user';
            const textPart = entry.parts.find(p => p.text);
            const content = textPart ? textPart.text : '';
            const imageParts = entry.parts.filter(p => p.inlineData);
            const images = imageParts.map(p => p.inlineData.data);

            const message = { role, content };
            if (images.length > 0) {
                message.images = images;
            }
            return message;
        }).filter(Boolean);

        const ollamaUrl = ollamaUrlInput.value.trim();
        const modelName = modelInput.value.trim();

        const requestBody = {
            model: modelName,
            messages: messagesForOllama,
            stream: false,
            // AIにファイル形式を伝えるためのシステムプロンプトを修正
            system: "You are a helpful AI assistant. Please respond to the user's latest message based on the context of the entire conversation history provided. Your response should be helpful, concise, and directly address the user's last query. If the user provides text from a file, it will be clearly marked with '--- START OF FILE: ... ---' and '--- END OF FILE: ... ---'."
        };

        const response = await fetch(`${ollamaUrl}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const responseText = await response.text();
            let errorMessage = `HTTP error! status: ${response.status} - ${responseText}`;
            try {
                const errorJson = JSON.parse(responseText);
                errorMessage = errorJson.error || errorMessage;
            } catch (e) {
                // JSONでなければ、そのままテキストをエラーメッセージとする
            }
            throw new Error(errorMessage);
        }

        const data = await response.json();
        const text = data.message.content;
        const aiTimestamp = new Date().toISOString();

        conversationHistory.push({ role: 'model', parts: [{ text }], timestamp: aiTimestamp });
        addMessageToDisplay('model', text, [], aiTimestamp);

        playNotificationSound();
        clearError();
    } catch (error) {
        console.error('API Error:', error);
        // On error, remove the user message that was optimistically added
        conversationHistory.pop();

        const msg = error.message.includes("fetch") || error.message.includes("Failed to fetch")
            ? `Ollamaサーバーへの接続に失敗しました。URL (${ollamaUrlInput.value}) とサーバーのCORS設定を確認してください。`
            : `APIエラー: ${error.message}`;
        displayError(msg);

        // Remove the visual representation of the failed user message
        if (chatHistory.lastChild && chatHistory.lastChild.classList.contains('message-row')) {
            chatHistory.removeChild(chatHistory.lastChild);
        }
    } finally {
        inputWrapper.classList.remove('submitting');
        sendButton.disabled = false; messageInput.disabled = false; fileInput.disabled = false;
        fileInputLabel.style.cssText = 'cursor: pointer; opacity: 1;';
        if (loadingIndicator.parentNode) loadingIndicator.parentNode.removeChild(loadingIndicator);
        attachedFiles = []; renderFilePreview(); messageInput.value = ''; autoResizeTextarea();
    }
}

function fileToGenerativePartForHistory(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve({ inlineData: { data: reader.result.split(',')[1], mimeType: file.type } });
        reader.onerror = reject; reader.readAsDataURL(file);
    });
}

async function handleSaveHistory() {
    try {
        if (conversationHistory.length <= 1) { showToast("保存する対話履歴がありません。"); return; }

        const dataToSave = {
            version: "2.0", // 互換性のあるバージョン番号
            model: modelInput.value.trim(),
            history: conversationHistory
        };

        const blob = new Blob([JSON.stringify(dataToSave, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        const now = new Date();
        a.download = `chat-history-${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}.json`;
        document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(a.href);
        showToast("履歴を保存しました。");
    } catch (error) { displayError("履歴の保存に失敗しました。"); }
}

function handleRestoreHistory(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const loadedData = JSON.parse(e.target.result);
            let historyForUI, timestampsForUI;

            if ((loadedData.version === "2.0" || loadedData.version === "1.0") && Array.isArray(loadedData.history)) {
                conversationHistory = loadedData.history;
                historyForUI = loadedData.history.map(entry => { const { timestamp, ...rest } = entry; return rest; });
                timestampsForUI = loadedData.history.map(entry => entry.timestamp);
                messageTimestamps = loadedData.history.map(entry => entry.timestamp);
            } else {
                throw new Error("無効な履歴ファイル形式です。");
            }

            if (loadedData.model) {
                modelInput.value = loadedData.model;
                updateAppTitle();
            }

            rebuildChatDisplay(historyForUI, timestampsForUI);

            settingsForm.classList.add('hidden');
            chatContainer.style.display = 'flex';
            clearError();
            showToast("履歴を復元しました。");
        } catch (error) {
            displayError(`履歴の復元に失敗しました: ${error.message}`);
        } finally {
            event.target.value = '';
        }
    };
    reader.onerror = () => displayError("ファイルの読み込み中にエラーが発生しました。"); reader.readAsText(file);
}

function rebuildChatDisplay(history, timestamps) {
    chatHistory.innerHTML = '';
    attachedFileObjects.clear();
    attachedFileIdCounter = 0;

    history.forEach((message, index) => {
        const timestamp = timestamps[index];
        if (!timestamp) return; // 念のため

        const role = message.role === 'model' ? 'ai' : 'user';
        const text = message.parts.find(p => p.text)?.text || '';
        const files = message.parts.filter(p => p.inlineData).map(part => ({
            inlineData: part.inlineData,
            name: `file_${index}`,
            isReconstructed: true
        }));
        addMessageToDisplay(role, text, files, timestamp);
    });

    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
        Promise.resolve().then(() => MathJax.typesetPromise([chatHistory])).catch((err) => console.error('MathJax error:', err));
    }
}

function downloadFile(file) {
    const url = URL.createObjectURL(file);
    const a = document.createElement('a');
    a.href = url;
    a.download = file.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function handleDownloadClick(code, language, messageDate) {
    const langMap = { 'javascript': 'js', 'python': 'py', 'html': 'html', 'css': 'css', 'java': 'java', 'csharp': 'cs', 'cpp': 'cpp', 'ruby': 'rb', 'go': 'go', 'shell': 'sh', 'bash': 'sh', 'json': 'json', 'sql': 'sql', 'typescript': 'ts', 'xml': 'xml', 'yaml': 'yaml', 'svg': 'svg' };
    const extension = langMap[language.toLowerCase()] || 'txt';
    const blob = new Blob([code], { type: 'text/plain' });

    const yyyy = messageDate.getFullYear();
    const mm = String(messageDate.getMonth() + 1).padStart(2, '0');
    const dd = String(messageDate.getDate()).padStart(2, '0');
    const h = String(messageDate.getHours()).padStart(2, '0');
    const min = String(messageDate.getMinutes()).padStart(2, '0');
    const fileName = `code-${yyyy}${mm}${dd}-${h}${min}.${extension}`;

    const file = new File([blob], fileName, { type: 'text/plain' });
    downloadFile(file);
}

function showToast(message) {
    toastNotification.textContent = message; toastNotification.classList.add('show');
    setTimeout(() => { toastNotification.classList.remove('show'); }, 3000);
}

function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

function handleFileDrop({ dataTransfer }) { if (inputWrapper.classList.contains('submitting')) return; addFiles(dataTransfer.files); }

function handleFileSelect(e) { if (fileInput.disabled) return; addFiles(e.target.files); e.target.value = ''; }

function addFiles(files) {
    for (const file of files) {
        if (!attachedFiles.some(f => f.name === file.name && f.size === file.size)) attachedFiles.push(file);
    }
    renderFilePreview();
}

function renderFilePreview() {
    filePreview.innerHTML = '';
    attachedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div'); fileItem.className = 'file-item';
        const fileName = document.createElement('span'); fileName.textContent = file.name;
        const removeBtn = document.createElement('span'); removeBtn.className = 'remove-file'; removeBtn.textContent = '×';
        removeBtn.onclick = () => { attachedFiles.splice(index, 1); renderFilePreview(); };
        fileItem.appendChild(fileName); fileItem.appendChild(removeBtn); filePreview.appendChild(fileItem);
    });
}

function displayError(message) { errorMessageDiv.innerHTML = message; errorMessageDiv.style.display = 'block'; }

function clearError() { errorMessageDiv.textContent = ''; errorMessageDiv.style.display = 'none'; }

function autoResizeTextarea() { messageInput.style.height = 'auto'; messageInput.style.height = `${messageInput.scrollHeight}px`; }

function handleKeyDown(e) { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); chatForm.requestSubmit(); } }

function handlePaste(e) {
    if (inputWrapper.classList.contains('submitting')) return;
    const clipboardData = e.clipboardData || window.clipboardData;
    if (!clipboardData) return;
    const items = clipboardData.items;
    const filesToPaste = [];
    for (let i = 0; i < items.length; i++) {
        if (items[i].kind === 'file' && items[i].type.startsWith('image/')) {
            const file = items[i].getAsFile();
            if (file) {
                const timestamp = new Date().getTime();
                const extension = file.type.split('/')[1] || 'png';
                const newName = `pasted-image-${timestamp}.${extension}`;
                const newFile = new File([file], newName, { type: file.type, lastModified: file.lastModified, });
                filesToPaste.push(newFile);
            }
        }
    }
    if (filesToPaste.length > 0) {
        e.preventDefault();
        addFiles(filesToPaste);
    }
}

function initializeApp() {
    if (typeof marked !== 'undefined') {
        marked.setOptions({ gfm: true, breaks: true, mangle: false, headerIds: false });
    }

    const initAudio = () => {
        if (audioContext) return;
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        } catch (e) { console.error("Web Audio API is not supported in this browser.", e); }
    };
    document.body.addEventListener('click', initAudio, { once: true });
    document.body.addEventListener('keydown', initAudio, { once: true });

    settingsForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const ollamaUrl = ollamaUrlInput.value.trim();
        const model = modelInput.value.trim();
        if (!ollamaUrl || !model) {
            displayError("OllamaサーバーURLとモデル名を両方入力してください。");
            return;
        }
        const newUrlParams = new URLSearchParams();
        newUrlParams.set('model', model);
        newUrlParams.set('ollamaUrl', ollamaUrl);
        window.location.search = newUrlParams.toString();
    });

    settingsToggleBtn.addEventListener('click', () => settingsForm.classList.toggle('hidden'));

    chatForm.addEventListener('submit', handleChatSubmit);
    messageInput.addEventListener('input', autoResizeTextarea);
    messageInput.addEventListener('keydown', handleKeyDown);
    fileInput.addEventListener('change', handleFileSelect);

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eName => inputWrapper.addEventListener(eName, preventDefaults, false));
    ['dragenter', 'dragover'].forEach(eName => inputWrapper.addEventListener(eName, () => inputWrapper.classList.add('drag-over'), false));
    ['dragleave', 'drop'].forEach(eName => inputWrapper.addEventListener(eName, () => inputWrapper.classList.remove('drag-over'), false));
    inputWrapper.addEventListener('drop', handleFileDrop, false);
    inputWrapper.addEventListener('paste', handlePaste, false);

    saveHistoryBtn.addEventListener('click', handleSaveHistory);
    restoreHistoryInput.addEventListener('change', handleRestoreHistory);

    chatHistory.addEventListener('click', (e) => {
        const codeButton = e.target.closest('.code-copy-btn, .code-download-btn');
        if (codeButton) {
            const wrapper = codeButton.closest('.code-block-wrapper');
            const blockId = wrapper?.dataset.blockId;
            const block = blockId ? codeBlocks.get(blockId) : null;
            if (!block) return;
            if (codeButton.classList.contains('code-copy-btn')) {
                navigator.clipboard.writeText(block.code).then(() => showToast("コードをコピーしました！"));
            } else {
                const messageDate = block.timestamp ? new Date(block.timestamp) : new Date();
                handleDownloadClick(block.code, block.language, messageDate);
            }
            return;
        }

        const msgCopyButton = e.target.closest('.copy-message-btn');
        if (msgCopyButton) {
            const textToCopy = msgCopyButton.dataset.copyText;
            if (textToCopy !== null && textToCopy !== undefined) {
                navigator.clipboard.writeText(textToCopy).then(() => {
                    showToast("メッセージをコピーしました！");
                }).catch(err => {
                    console.error('Failed to copy text:', err);
                    showToast("コピーに失敗しました。");
                });
            }
            return;
        }

        const fileElement = e.target.closest('[data-file-id]');
        if(fileElement) {
            const fileId = fileElement.dataset.fileId;
            const fileToDownload = attachedFileObjects.get(fileId);
            if (fileToDownload) {
                downloadFile(fileToDownload);
            }
        }
    });

    loadingIndicator = document.createElement('div');
    loadingIndicator.id = 'loading-indicator';
    loadingIndicator.innerHTML = `
        <div class="message-row ai">
            <div class="typing-indicator-bubble">
                <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
            </div>
            <div class="message-timestamp">&nbsp;</div>
        </div>
    `;

    const urlParams = new URLSearchParams(window.location.search);
    const ollamaUrlFromUrl = urlParams.get('ollamaUrl');
    const modelFromUrl = urlParams.get('model');
    debugMode = urlParams.get('debugMode');

    ollamaUrlInput.value = ollamaUrlFromUrl || 'http://localhost:11434';
    modelInput.value = modelFromUrl || 'llama3';
    updateAppTitle();

    if (debugMode === 'user' || debugMode === 'ai') {
        settingsForm.classList.add('hidden');
        chatContainer.style.display = 'flex';
        clearError();
        displayInitialWelcomeMessage();

        const debugNotice = document.createElement('div');
        debugNotice.style.cssText = 'background-color: #fff3cd; color: #856404; padding: 10px 20px; text-align: center; font-weight: bold; border-bottom: 1px solid var(--border-color); font-size: 0.9em;';
        debugNotice.textContent = `デバッグモード (${debugMode}) で動作中`;
        appContainer.insertBefore(debugNotice, settingsForm.nextSibling);

        const historyManagement = document.querySelector('.history-management');
        if (historyManagement) {
            historyManagement.style.opacity = '0.5';
            historyManagement.style.pointerEvents = 'none';
            const historyButtons = historyManagement.querySelectorAll('button, label');
            historyButtons.forEach(btn => btn.tabIndex = -1);
        }

    } else if (ollamaUrlFromUrl) {
        settingsForm.classList.add('hidden');
        chatContainer.style.display = 'flex';
        clearError();
        displayInitialWelcomeMessage();
    } else {
        settingsForm.classList.remove('hidden');
        displayError("Ollamaサーバーが設定されていません。設定画面でURLとモデル名を入力し、「適用」を押してください。");
        chatContainer.style.display = 'none';
    }
}

// Initialize the application
initializeApp();