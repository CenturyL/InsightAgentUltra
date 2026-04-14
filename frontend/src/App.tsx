import { useEffect, useMemo, useRef, useState } from "react";
import { marked } from "marked";

type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  thought: string;
  content: string;
  traces: string[];
  createdAt: number;
};

type PaePlanStep = {
  stepId: string;
  capability: string;
  goal: string;
  status?: string;
};

type PaeExecutionStep = {
  stepId: string;
  status: string;
  goal: string;
};

type ParsedPaeTrace = {
  active: boolean;
  entered: boolean;
  planning: boolean;
  reflecting: boolean;
  generating: boolean;
  completed: boolean;
  currentStepId: string | null;
  planSteps: PaePlanStep[];
  executionSteps: PaeExecutionStep[];
  reflectionSteps: Array<{ stepId: string; status: string }>;
  activatedSkills: string[];
  allowedTools: string[];
  mcpToolCalls: string[];
  raw: string[];
};

type CombinedPaeStep = {
  stepId: string;
  capability: string;
  goal: string;
  status: string;
};

type UploadState = {
  filename: string;
  chunksInserted: number;
  source: string;
} | null;

type RuntimeSkillAsset = {
  filename: string;
  content: string;
  source: string;
};

type RuntimeAssets = {
  agents_md: string;
  soul_md: string;
  memory_md: string;
  skills: RuntimeSkillAsset[];
};

type RuntimeMCPServer = {
  server_name: string;
  transport: string;
  command?: string | null;
  args?: string[];
  cwd?: string | null;
  url?: string | null;
  headers?: Record<string, string>;
};

type RuntimeMCPStatus = {
  server_name: string;
  transport: string;
  connected: boolean;
  tool_names: string[];
};

type RuntimeMCPConfig = {
  config_text: string;
  servers: RuntimeMCPServer[];
  status: RuntimeMCPStatus[];
};

type SessionSummary = {
  thread_id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_message_preview: string;
};

type SessionBootstrapResponse = {
  sessions: SessionSummary[];
  current_thread_id: string;
};

type SessionMessagesResponse = {
  thread_id: string;
  messages: Array<{ role: string; content: string }>;
};

type PromptPreset = {
  label: string;
  prompt: string;
};

const TRACE_PREFIXES = [
  ">",
  "🧠",
  "🧭",
  "📝",
  "🛠️",
  "🔁",
  "📎",
  "✍️",
  "✅",
  "⚠️",
  "❌",
  "💾",
  "🌐",
  "🔎",
  "⏱️",
  "🧩",
  "🧰",
  "🤖",
];

const PROMPT_PRESETS: PromptPreset[] = [
  {
    label: "测试 MCP",
    prompt: "请列出当前项目根目录下有哪些一级文件和文件夹，只使用 MCP 工具完成，不要猜测。",
  },
  {
    label: "测试 MCP+SKILL",
    prompt: "请抓取 https://modelcontextprotocol.io 的首页，告诉我这个网站主要是什么，只使用 MCP 工具，不要用你已有知识补充。",
  },
  {
    label: "测试 PAE",
    prompt: "请比较这个项目里的 skills/ 和 .claude/skills/ 两类 skill 的区别，按表格输出。",
  },
];

function uid(): string {
  const randomUUID = globalThis.crypto?.randomUUID?.bind(globalThis.crypto);
  if (randomUUID) return randomUUID();
  const randomPart = Math.random().toString(36).slice(2, 10);
  return `id-${Date.now()}-${randomPart}`;
}

function getSkillDisplayName(filename: string): string {
  const normalized = filename.replace(/\\/g, "/");
  const match = normalized.match(/([^/]+)\/SKILL\.md$/i);
  if (match?.[1]) return match[1];
  return normalized.replace(/\/SKILL\.md$/i, "").split("/").pop() || normalized;
}

function getSkillKey(skill: RuntimeSkillAsset): string {
  return `${skill.source}:${skill.filename}`;
}

function isTraceBlock(text: string) {
  return TRACE_PREFIXES.some((prefix) => text.startsWith(prefix)) || /^-+\s*step[_-]?\d+/i.test(text);
}

function renderMarkdown(content: string) {
  return { __html: marked.parse(content) as string };
}

function uniqueTextBlocks(blocks: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;
    const normalized = trimmed.replace(/\s+/g, " ");
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    result.push(trimmed);
  }
  return result;
}

function createAssistantPlaceholder(): ChatMessage {
  return {
    id: uid(),
    role: "assistant",
    thought: "",
    content: "",
    traces: [],
    createdAt: Date.now(),
  };
}

function parsePaeTrace(traces: string[], content = ""): ParsedPaeTrace {
  const parsed: ParsedPaeTrace = {
    active: false,
    entered: false,
    planning: false,
    reflecting: false,
    generating: false,
    completed: false,
    currentStepId: null,
    planSteps: [],
    executionSteps: [],
    reflectionSteps: [],
    activatedSkills: [],
    allowedTools: [],
    mcpToolCalls: [],
    raw: traces,
  };

  for (const trace of traces) {
    if (trace.includes("[PAE调用]") || trace.includes("[计划模式]")) {
      parsed.active = true;
      parsed.entered = true;
      continue;
    }
    if (trace.startsWith("📝 [任务规划]")) {
      parsed.active = true;
      parsed.planning = true;
      continue;
    }
    if (/^-+\s*step[_-]?\d+/i.test(trace)) {
      const [, stepId = "", capability = "", goal = ""] = trace.split("|").map((item) => item.trim().replace(/^- /, ""));
      if (stepId) {
        parsed.planSteps.push({ stepId, capability, goal });
        parsed.active = true;
      }
      continue;
    }
    if (trace.startsWith("🛠️ [步骤开始]")) {
      const [, payload = ""] = trace.split("] ");
      const [stepId = ""] = payload.split("|").map((item) => item.trim());
      parsed.currentStepId = stepId || null;
      parsed.active = true;
      continue;
    }
    if (trace.startsWith("🛠️ [步骤执行]")) {
      const [, payload = ""] = trace.split("] ");
      const [stepId = "", status = "", goal = ""] = payload.split("|").map((item) => item.trim());
      parsed.executionSteps.push({ stepId, status, goal });
      if (parsed.currentStepId === stepId) {
        parsed.currentStepId = null;
      }
      parsed.active = true;
      continue;
    }
    if (trace.startsWith("🔁 [反思修正]")) {
      parsed.active = true;
      parsed.reflecting = true;
      continue;
    }
    if (trace.startsWith("🧩 [Skill激活]")) {
      const [, payload = ""] = trace.split("] ");
      parsed.activatedSkills = payload
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      continue;
    }
    if (trace.startsWith("🧰 [工具约束]")) {
      const [, payload = ""] = trace.split("] ");
      parsed.allowedTools = payload
        .split("、")
        .map((item) => item.trim())
        .filter(Boolean);
      continue;
    }
    if (trace.startsWith("🧩 [MCP工具调用]")) {
      const matched = trace.match(/【(.+?)】/);
      if (matched?.[1]) {
        parsed.mcpToolCalls.push(matched[1]);
      }
      continue;
    }
    if (trace.startsWith("📎 [反思结果]")) {
      const [, payload = ""] = trace.split("] ");
      const [stepId = "", status = ""] = payload.split("|").map((item) => item.trim());
      parsed.reflectionSteps.push({ stepId, status });
      parsed.active = true;
      continue;
    }
    if (trace.startsWith("✍️ [答案生成]")) {
      parsed.active = true;
      parsed.generating = true;
      continue;
    }
    if (trace.startsWith("✅ [PAE完成]")) {
      parsed.active = true;
      parsed.completed = true;
    }
  }

  if (content.includes("已进入 Plan-and-Execute 子流程")) {
    parsed.active = true;
    parsed.entered = true;
  }
  if (content.includes("已生成执行计划")) {
    parsed.active = true;
    parsed.planning = true;
  }
  if (content.includes("正在生成最终结果")) {
    parsed.active = true;
    parsed.generating = true;
  }
  if (content.includes("Plan-and-Execute 子流程执行完成")) {
    parsed.active = true;
    parsed.completed = true;
  }

  if (parsed.planSteps.length > 0) {
    parsed.active = true;
    parsed.entered = true;
    parsed.planning = true;
  }
  if (parsed.executionSteps.length > 0) {
    parsed.active = true;
  }
  if (parsed.reflectionSteps.length > 0) {
    parsed.active = true;
    parsed.reflecting = true;
  }
  if (parsed.completed && parsed.planSteps.length > 0) {
    parsed.entered = true;
    parsed.planning = true;
    parsed.reflecting = true;
    if (parsed.executionSteps.length === 0) {
      parsed.executionSteps = parsed.planSteps.map((step) => ({
        stepId: step.stepId,
        status: "completed",
        goal: step.goal,
      }));
    }
    if (parsed.reflectionSteps.length === 0) {
      parsed.reflectionSteps = parsed.planSteps.map((step) => ({
        stepId: step.stepId,
        status: "completed",
      }));
    }
    parsed.generating = true;
  }

  return parsed;
}

function buildCombinedPaeSteps(parsed: ParsedPaeTrace): CombinedPaeStep[] {
  const executionByStep = new Map(parsed.executionSteps.map((step) => [step.stepId, step]));
  return parsed.planSteps.map((step) => ({
    stepId: step.stepId,
    capability: step.capability,
    goal: step.goal,
    status: executionByStep.get(step.stepId)?.status ?? "pending",
  }));
}

export default function App() {
  const apiBase = "/api/v3";
  const [userIdInput, setUserIdInput] = useState("");
  const [userId, setUserId] = useState("");
  const [currentThreadId, setCurrentThreadId] = useState("");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [modelChoice, setModelChoice] = useState("deepseek_chat");
  const [forcePlan, setForcePlan] = useState(false);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadState, setUploadState] = useState<UploadState>(null);
  const [runtimeAssets, setRuntimeAssets] = useState<RuntimeAssets>({
    agents_md: "",
    soul_md: "",
    memory_md: "",
    skills: [],
  });
  const [selectedSkillFilename, setSelectedSkillFilename] = useState("");
  const [mcpConfig, setMcpConfig] = useState<RuntimeMCPConfig>({
    config_text: "",
    servers: [],
    status: [],
  });
  const [assetsLoading, setAssetsLoading] = useState(false);
  const [assetsSaving, setAssetsSaving] = useState(false);
  const [mcpLoading, setMcpLoading] = useState(false);
  const [mcpSaving, setMcpSaving] = useState(false);
  const [error, setError] = useState("");
  const [toastMessage, setToastMessage] = useState("");
  const [activeTraceMessageId, setActiveTraceMessageId] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(false);
  const [showTracePanel, setShowTracePanel] = useState(false);
  const [expandedThoughtIds, setExpandedThoughtIds] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const messagesRef = useRef<HTMLElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const toastTimerRef = useRef<number | null>(null);
  const shouldAutoScrollRef = useRef(true);

  const activeTraceMessage = useMemo(
    () =>
      messages.find((message) => message.id === activeTraceMessageId) ??
      [...messages].reverse().find((message) => message.role === "assistant") ??
      null,
    [activeTraceMessageId, messages],
  );
  const normalizedUserIdInput = userIdInput.trim();
  const isConfirmedUserId = Boolean(userId.trim()) && userId.trim() === normalizedUserIdInput;
  const activePaeTrace = useMemo(
    () => parsePaeTrace(activeTraceMessage?.traces ?? [], activeTraceMessage?.content ?? ""),
    [activeTraceMessage],
  );
  const combinedPaeSteps = useMemo(
    () => buildCombinedPaeSteps(activePaeTrace),
    [activePaeTrace],
  );

  useEffect(() => {
    if (!shouldAutoScrollRef.current) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isSending]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [query]);

  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth > 960) {
        setShowSidebar(false);
        setShowTracePanel(false);
      }
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    void loadRuntimeAssets();
    void loadRuntimeMcpConfig();
  }, []);

  useEffect(() => {
    if (!runtimeAssets.skills.length) {
      if (selectedSkillFilename) setSelectedSkillFilename("");
      return;
    }
    if (!runtimeAssets.skills.some((skill) => getSkillKey(skill) === selectedSkillFilename)) {
      setSelectedSkillFilename(getSkillKey(runtimeAssets.skills[0]));
    }
  }, [runtimeAssets.skills, selectedSkillFilename]);

  useEffect(() => {
    return () => {
      if (toastTimerRef.current) {
        window.clearTimeout(toastTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!runtimeAssets.skills.length) {
      setSelectedSkillFilename("");
      return;
    }
    if (!runtimeAssets.skills.some((skill) => getSkillKey(skill) === selectedSkillFilename)) {
      setSelectedSkillFilename(getSkillKey(runtimeAssets.skills[0]));
    }
  }, [runtimeAssets.skills, selectedSkillFilename]);

  function showToast(message: string) {
    setToastMessage(message);
    if (toastTimerRef.current) {
      window.clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = window.setTimeout(() => {
      setToastMessage("");
      toastTimerRef.current = null;
    }, 2200);
  }

  function setThoughtExpanded(messageId: string, expanded: boolean) {
    setExpandedThoughtIds((prev) => {
      const has = prev.includes(messageId);
      if (expanded && !has) return [...prev, messageId];
      if (!expanded && has) return prev.filter((id) => id !== messageId);
      return prev;
    });
  }

  function updateAutoScrollLock() {
    const el = messagesRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    shouldAutoScrollRef.current = distanceFromBottom <= 96;
  }

  function hydrateSessionMessages(items: Array<{ role: string; content: string }>): ChatMessage[] {
    const base = Date.now();
    return items.map((item, index) => ({
      id: uid(),
      role: item.role === "user" ? "user" : "assistant",
      thought: "",
      content: item.content ?? "",
      traces: [],
      createdAt: base + index,
    }));
  }

  function formatSessionTime(value: string): string {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  }

  function applyPromptPreset(prompt: string) {
    setQuery(prompt);
    window.requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
  }

  async function loadRuntimeAssets() {
    setAssetsLoading(true);
    try {
      const response = await fetch(`${apiBase}/runtime/assets`);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = (await response.json()) as RuntimeAssets;
      setRuntimeAssets({
        agents_md: data.agents_md ?? "",
        soul_md: data.soul_md ?? "",
        memory_md: data.memory_md ?? "",
        skills: Array.isArray(data.skills)
          ? data.skills.map((skill) => ({
              filename: skill.filename,
              content: skill.content,
              source: skill.source ?? "project",
            }))
          : [],
      });
      if (!selectedSkillFilename && Array.isArray(data.skills) && data.skills.length > 0) {
        const firstSkill = {
          filename: data.skills[0].filename,
          content: data.skills[0].content,
          source: data.skills[0].source ?? "project",
        };
        setSelectedSkillFilename(getSkillKey(firstSkill));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "运行时资产加载失败");
    } finally {
      setAssetsLoading(false);
    }
  }

  async function refreshSessions(nextUserId = userId) {
    const normalized = nextUserId.trim();
    if (!normalized) return;
    const response = await fetch(`${apiBase}/sessions?user_id=${encodeURIComponent(normalized)}`);
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = (await response.json()) as { sessions: SessionSummary[] };
    setSessions(Array.isArray(data.sessions) ? data.sessions : []);
  }

  async function confirmUserSessions() {
    const normalized = userIdInput.trim();
    if (!normalized || sessionsLoading) return;
    setSessionsLoading(true);
    setError("");
    try {
      const response = await fetch(`${apiBase}/sessions/bootstrap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: normalized }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = (await response.json()) as SessionBootstrapResponse;
      setUserId(normalized);
      setCurrentThreadId(data.current_thread_id);
      setSessions(Array.isArray(data.sessions) ? data.sessions : []);
      setMessages([]);
      setActiveTraceMessageId(null);
      shouldAutoScrollRef.current = true;
      showToast("已加载历史会话，并创建了新的空会话。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "历史会话加载失败");
    } finally {
      setSessionsLoading(false);
    }
  }

  async function createNewSession() {
    if (!userId.trim() || sessionsLoading) {
      showToast("请先确认 USERID");
      return;
    }
    setSessionsLoading(true);
    setError("");
    try {
      const response = await fetch(`${apiBase}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId.trim() }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const created = (await response.json()) as SessionSummary;
      setCurrentThreadId(created.thread_id);
      setMessages([]);
      setActiveTraceMessageId(null);
      await refreshSessions(userId.trim());
      shouldAutoScrollRef.current = true;
    } catch (err) {
      setError(err instanceof Error ? err.message : "新会话创建失败");
    } finally {
      setSessionsLoading(false);
    }
  }

  async function openSession(threadId: string) {
    if (!userId.trim() || sessionsLoading || !threadId) return;
    setSessionsLoading(true);
    setError("");
    try {
      const response = await fetch(
        `${apiBase}/sessions/${encodeURIComponent(threadId)}/messages?user_id=${encodeURIComponent(userId.trim())}`,
      );
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = (await response.json()) as SessionMessagesResponse;
      setCurrentThreadId(data.thread_id);
      setMessages(hydrateSessionMessages(Array.isArray(data.messages) ? data.messages : []));
      setActiveTraceMessageId(null);
      shouldAutoScrollRef.current = true;
    } catch (err) {
      setError(err instanceof Error ? err.message : "历史会话加载失败");
    } finally {
      setSessionsLoading(false);
    }
  }

  async function deleteCurrentSession(threadId: string) {
    if (!userId.trim() || sessionsLoading || !threadId) return;
    setSessionsLoading(true);
    setError("");
    try {
      const response = await fetch(
        `${apiBase}/sessions/${encodeURIComponent(threadId)}?user_id=${encodeURIComponent(userId.trim())}`,
        { method: "DELETE" },
      );
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const nextSessions = sessions.filter((session) => session.thread_id !== threadId);
      setSessions(nextSessions);
      if (currentThreadId === threadId) {
        if (nextSessions[0]) {
          await openSession(nextSessions[0].thread_id);
        } else {
          const created = await fetch(`${apiBase}/sessions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: userId.trim() }),
          });
          if (!created.ok) {
            throw new Error(await created.text());
          }
          const createdSession = (await created.json()) as SessionSummary;
          setCurrentThreadId(createdSession.thread_id);
          setMessages([]);
          setActiveTraceMessageId(null);
          await refreshSessions(userId.trim());
        }
      } else {
        await refreshSessions(userId.trim());
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除会话失败");
    } finally {
      setSessionsLoading(false);
    }
  }

  async function loadRuntimeMcpConfig() {
    setMcpLoading(true);
    try {
      const response = await fetch(`${apiBase}/runtime/mcp/config`);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = (await response.json()) as RuntimeMCPConfig;
      setMcpConfig({
        config_text: data.config_text ?? "",
        servers: Array.isArray(data.servers) ? data.servers : [],
        status: Array.isArray(data.status) ? data.status : [],
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "MCP 配置加载失败");
    } finally {
      setMcpLoading(false);
    }
  }

  async function saveRuntimeAssets() {
    setAssetsSaving(true);
    setError("");
    try {
      const response = await fetch(`${apiBase}/runtime/assets`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(runtimeAssets),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = (await response.json()) as RuntimeAssets;
      setRuntimeAssets({
        agents_md: data.agents_md ?? "",
        soul_md: data.soul_md ?? "",
        memory_md: data.memory_md ?? "",
        skills: Array.isArray(data.skills)
          ? data.skills.map((skill) => ({
              filename: skill.filename,
              content: skill.content,
              source: skill.source ?? "project",
            }))
          : [],
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "运行时资产保存失败");
    } finally {
      setAssetsSaving(false);
    }
  }

  async function saveRuntimeMcpConfig() {
    setMcpSaving(true);
    setError("");
    try {
      const response = await fetch(`${apiBase}/runtime/mcp/config`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ config_text: mcpConfig.config_text }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = (await response.json()) as RuntimeMCPConfig;
      setMcpConfig(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "MCP 配置保存失败");
    } finally {
      setMcpSaving(false);
    }
  }

  async function reloadRuntimeMcpConfig() {
    setMcpSaving(true);
    setError("");
    try {
      const response = await fetch(`${apiBase}/runtime/mcp/reload`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = (await response.json()) as RuntimeMCPConfig;
      setMcpConfig(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "MCP 重载失败");
    } finally {
      setMcpSaving(false);
    }
  }

  async function handleSkillUpload(files: FileList | null) {
    if (!files?.length) return;
    setAssetsSaving(true);
    setError("");
    try {
      const uploaded = await Promise.all(
        Array.from(files).map(async (file) => ({
          filename: file.name,
          content: await file.text(),
          source: "project",
        })),
      );

      const mergedMap = new Map(runtimeAssets.skills.map((skill) => [getSkillKey(skill), skill]));
      for (const skill of uploaded) {
        mergedMap.set(getSkillKey(skill), skill);
      }
      const nextAssets: RuntimeAssets = {
        ...runtimeAssets,
        skills: Array.from(mergedMap.values()).sort((a, b) => a.filename.localeCompare(b.filename)),
      };

      const response = await fetch(`${apiBase}/runtime/assets`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(nextAssets),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = (await response.json()) as RuntimeAssets;
      setRuntimeAssets({
        agents_md: data.agents_md ?? "",
        soul_md: data.soul_md ?? "",
        memory_md: data.memory_md ?? "",
        skills: Array.isArray(data.skills)
          ? data.skills.map((skill) => ({
              filename: skill.filename,
              content: skill.content,
              source: skill.source ?? "project",
            }))
          : [],
      });
      if (!selectedSkillFilename && data.skills.length > 0) {
        const firstSkill = {
          filename: data.skills[0].filename,
          content: data.skills[0].content,
          source: data.skills[0].source ?? "project",
        };
        setSelectedSkillFilename(getSkillKey(firstSkill));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "技能上传失败");
    } finally {
      setAssetsSaving(false);
    }
  }

  function updateSkillContent(skillKey: string, content: string) {
    setRuntimeAssets((prev) => ({
      ...prev,
      skills: prev.skills.map((skill) => (getSkillKey(skill) === skillKey ? { ...skill, content } : skill)),
    }));
  }

  function addSkillDraft() {
    const filename = `new-skill/SKILL.md`;
    setRuntimeAssets((prev) => {
      if (prev.skills.some((skill) => getSkillKey(skill) === `project:${filename}`)) return prev;
      return {
        ...prev,
        skills: [
          ...prev.skills,
          {
            filename,
            content:
              "---\nname: new-skill\ndescription: Describe when this skill should be used.\nuser-invocable: true\n---\n\n# New Skill\n\nWrite the skill instructions here.\n",
            source: "project",
          },
        ].sort((a, b) => a.filename.localeCompare(b.filename)),
      };
    });
    setSelectedSkillFilename(`project:${filename}`);
  }

  function deleteSelectedSkill() {
    if (!selectedSkill) return;
    setRuntimeAssets((prev) => ({
      ...prev,
      skills: prev.skills.filter((skill) => getSkillKey(skill) !== getSkillKey(selectedSkill)),
    }));
  }

  const selectedSkill = runtimeAssets.skills.find((skill) => getSkillKey(skill) === selectedSkillFilename) ?? runtimeAssets.skills[0] ?? null;

  async function handleUpload(file: File) {
    setUploading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${apiBase}/knowledge/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = await response.json();
      setUploadState({
        filename: data.filename ?? file.name,
        chunksInserted: data.chunks_inserted ?? 0,
        source: data.source ?? file.name,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setUploading(false);
    }
  }

  async function handleSend() {
    if (!userId.trim()) {
      showToast("请先确认 USERID");
      return;
    }
    if (!currentThreadId.trim()) {
      showToast("请先创建或选择一个会话");
      return;
    }
    if (!query.trim() || isSending) return;

    setError("");
    setIsSending(true);

    const userMessage: ChatMessage = {
      id: uid(),
      role: "user",
      thought: "",
      content: query.trim(),
      traces: [],
      createdAt: Date.now(),
    };
    const assistantPlaceholder = createAssistantPlaceholder();
    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setActiveTraceMessageId(assistantPlaceholder.id);
    setQuery("");
    shouldAutoScrollRef.current = true;

    try {
      const response = await fetch(`${apiBase}/chat/agent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: userMessage.content,
          thread_id: currentThreadId,
          user_id: userId.trim(),
          plan_mode: forcePlan ? "strict_plan" : "auto",
          model_choice: modelChoice,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(await response.text());
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      const pushEvent = (rawLine: string) => {
        const trimmed = rawLine.trim();
        if (!trimmed) return;
        let payload: { type?: string; content?: string } = {};
        try {
          payload = JSON.parse(trimmed);
        } catch {
          return;
        }
        const type = payload.type || "answer";
        const content = payload.content || "";
        if (!content) return;
        setMessages((prev) =>
          prev.map((message) => {
            if (message.id !== assistantPlaceholder.id) return message;
            if (type === "trace") {
              if (message.traces[message.traces.length - 1] === content) return message;
              return { ...message, traces: [...message.traces, content] };
            }
            if (type === "thought") {
              return {
                ...message,
                thought: message.thought ? `${message.thought}${content}` : content,
              };
            }
            return {
              ...message,
              content: message.content ? `${message.content}${content}` : content,
            };
          }),
        );
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) pushEvent(line);
      }

      const finalChunk = buffer.trim();
      if (finalChunk) pushEvent(finalChunk);
      await refreshSessions(userId.trim());
    } catch (err) {
      const message = err instanceof Error ? err.message : "请求失败";
      setError(message);
      setMessages((prev) =>
        prev.map((item) =>
          item.id === assistantPlaceholder.id
            ? {
                ...item,
                traces: [...item.traces, `❌ 请求失败：${message}`],
                content: item.content || "请求失败，请检查后端服务或重试。",
              }
            : item,
        ),
      );
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${showSidebar ? "open" : ""}`}>
        <div className="brand sidebar-brand">
          <div className="brand-badge">IA</div>
          <div>
            <div className="brand-title">InsightAgentMax</div>
            <div className="brand-subtitle">Agent Runtime Studio</div>
          </div>
          <button className="drawer-close" type="button" onClick={() => setShowSidebar(false)}>
            关闭
          </button>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-title">Sessions</div>
          <div className="asset-card session-identity-card">
            <label className="field compact session-identity-field">
              <div className="session-identity-label-row">
                <span>UserID</span>
                <button
                  className="session-create-button session-create-button-inline"
                  type="button"
                  onClick={() => void createNewSession()}
                  disabled={!userId.trim() || sessionsLoading}
                >
                  新会话
                </button>
              </div>
              <div className="session-identity-note">输入 User ID 开启历史会话</div>
              <input
                value={userIdInput}
                onChange={(e) => setUserIdInput(e.target.value)}
                placeholder="输入 UserID 后点击确定"
                className={!userId.trim() ? "required" : ""}
              />
            </label>
            <button
              className={`primary-button sidebar-save-button sidebar-save-button-compact session-confirm-button ${
                isConfirmedUserId ? "confirmed" : ""
              }`}
              type="button"
              onClick={() => void confirmUserSessions()}
              disabled={!userIdInput.trim() || sessionsLoading}
            >
              {sessionsLoading ? "处理中..." : "确定"}
            </button>
            <div className="sidebar-title session-identity-header">History Sessions</div>
            <div className="session-list">
              {sessions.map((session) => (
                <div
                  key={session.thread_id}
                  className={`session-item ${currentThreadId === session.thread_id ? "active" : ""}`}
                >
                  <button type="button" className="session-item-open" onClick={() => void openSession(session.thread_id)}>
                    <div className="session-item-top">
                      <strong>{session.title}</strong>
                      <span>{formatSessionTime(session.updated_at)}</span>
                    </div>
                    <div className="session-item-preview">{session.last_message_preview || "空会话"}</div>
                  </button>
                  <div className="session-item-actions">
                    <button
                      type="button"
                      className="session-delete-button"
                      onClick={() => void deleteCurrentSession(session.thread_id)}
                      disabled={sessionsLoading}
                    >
                      删除
                    </button>
                  </div>
                </div>
              ))}
              {!sessions.length ? (
                <div className="skill-empty">{sessionsLoading ? "加载中..." : "确认 USERID 后显示历史会话"}</div>
              ) : null}
            </div>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-title sidebar-title-compact">Persona &amp; MD Memory</div>
          <div className="asset-card">
            <label className="field compact">
              <span>AGENTS.md</span>
              <textarea
                rows={4}
                className="runtime-asset-textarea"
                value={runtimeAssets.agents_md}
                onChange={(e) => setRuntimeAssets((prev) => ({ ...prev, agents_md: e.target.value }))}
                placeholder={assetsLoading ? "加载中..." : "AGENTS.md"}
              />
            </label>
            <label className="field compact">
              <span>SOUL.md</span>
              <textarea
                rows={4}
                className="runtime-asset-textarea"
                value={runtimeAssets.soul_md}
                onChange={(e) => setRuntimeAssets((prev) => ({ ...prev, soul_md: e.target.value }))}
                placeholder={assetsLoading ? "加载中..." : "SOUL.md"}
              />
            </label>
            <label className="field compact">
              <span>MEMORY.md</span>
              <textarea
                rows={5}
                className="runtime-asset-textarea"
                value={runtimeAssets.memory_md}
                onChange={(e) => setRuntimeAssets((prev) => ({ ...prev, memory_md: e.target.value }))}
                placeholder={assetsLoading ? "加载中..." : "MEMORY.md"}
              />
            </label>
            <button
              className="secondary-button sidebar-save-button sidebar-save-button-compact sidebar-action-button"
              type="button"
              onClick={() => void saveRuntimeAssets()}
              disabled={assetsLoading || assetsSaving}
            >
              {assetsSaving ? "保存中..." : "Save Runtime Setup"}
            </button>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-title">Skills</div>
          <div className="asset-card">
            <div className="skill-section skill-section-top">
              <div className="skill-chip-list skill-grid-list">
                {runtimeAssets.skills.map((skill) => (
                  <button
                    key={getSkillKey(skill)}
                    type="button"
                    className={`skill-chip skill-chip-button ${selectedSkill && getSkillKey(selectedSkill) === getSkillKey(skill) ? "active" : ""}`}
                    onClick={() => setSelectedSkillFilename(getSkillKey(skill))}
                    title={skill.filename}
                  >
                    {getSkillDisplayName(skill.filename)}
                  </button>
                ))}
                {!runtimeAssets.skills.length ? <span className="skill-empty">暂无已加载 Skills</span> : null}
              </div>
            </div>
            <div className="skill-section skill-section-middle">
              {selectedSkill ? (
                <label className="field compact">
                  <span>{getSkillDisplayName(selectedSkill.filename)}</span>
                  <textarea
                    rows={12}
                    className="runtime-asset-textarea skill-body-editor"
                    value={selectedSkill.content}
                    onChange={(e) => updateSkillContent(getSkillKey(selectedSkill), e.target.value)}
                  />
                </label>
              ) : (
                <div className="skill-empty">选择或上传一个 Skill package。</div>
              )}
            </div>
            <div className="skill-section skill-section-bottom">
              <div className="skill-toolbar skill-toolbar-bottom">
                <button
                  className="secondary-button sidebar-save-button-compact"
                  type="button"
                  onClick={addSkillDraft}
                >
                  新建 Skill
                </button>
                <button
                  className="secondary-button sidebar-save-button-compact skill-delete-button"
                  type="button"
                  onClick={deleteSelectedSkill}
                  disabled={!selectedSkill}
                >
                  删除 Skill
                </button>
                <label className="upload-box compact skill-upload-box sidebar-action-button">
                  <input
                    type="file"
                    accept=".md,text/markdown,text/plain"
                    multiple
                    onChange={(e) => {
                      void handleSkillUpload(e.target.files);
                      e.currentTarget.value = "";
                    }}
                  />
                  <span>{assetsSaving ? "上传中..." : "上传 SKILL.md"}</span>
                </label>
              </div>
            </div>
            <button
              className="secondary-button sidebar-save-button sidebar-save-button-compact sidebar-action-button"
              type="button"
              onClick={() => void saveRuntimeAssets()}
              disabled={assetsLoading || assetsSaving}
            >
              {assetsSaving ? "保存中..." : "保存 Skills"}
            </button>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-title">MCP</div>
          <div className="asset-card">
            <div className="mcp-status-list">
              {mcpConfig.status.map((item) => (
                <div key={item.server_name} className="mcp-status-card">
                  <div className="mcp-status-top">
                    <strong>{item.server_name}</strong>
                    <span className={item.connected ? "mcp-connected" : "mcp-disconnected"}>
                      {item.connected ? "connected" : "disconnected"}
                    </span>
                  </div>
                  <div className="mcp-status-meta">{item.transport}</div>
                  <div className="mcp-tool-list">
                    {item.tool_names.map((toolName) => (
                      <span key={toolName} className="skill-chip">
                        {toolName}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
              {!mcpConfig.status.length ? <span className="skill-empty">{mcpLoading ? "加载中..." : "暂无已连接 MCP server"}</span> : null}
            </div>
            <label className="field compact">
              <span>.mcp.json</span>
              <textarea
                rows={12}
                className="runtime-asset-textarea"
                value={mcpConfig.config_text}
                onChange={(e) => setMcpConfig((prev) => ({ ...prev, config_text: e.target.value }))}
                placeholder={mcpLoading ? "加载中..." : "{\n  \"mcpServers\": {}\n}"}
              />
            </label>
            <div className="mcp-toolbar">
              <label className="upload-box compact skill-upload-box sidebar-action-button">
                <input
                  type="file"
                  accept=".json,application/json"
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    const text = await file.text();
                    setMcpConfig((prev) => ({ ...prev, config_text: text }));
                    e.currentTarget.value = "";
                  }}
                />
                <span>上传本地 .mcp.json 文件</span>
              </label>
              <button
                className="secondary-button sidebar-save-button-compact"
                type="button"
                onClick={() => void reloadRuntimeMcpConfig()}
                disabled={mcpSaving}
              >
                {mcpSaving ? "处理中..." : "Reload MCP"}
              </button>
            </div>
            <button
              className="secondary-button sidebar-save-button sidebar-save-button-compact sidebar-action-button"
              type="button"
              onClick={() => void saveRuntimeMcpConfig()}
              disabled={mcpLoading || mcpSaving}
            >
              {mcpSaving ? "保存中..." : "保存 MCP 配置"}
            </button>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-title">Knowledge Base</div>
          <div className="asset-card">
            <label className="upload-box compact upload-box-small sidebar-action-button">
              <input
                type="file"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void handleUpload(file);
                }}
              />
              <span>{uploading ? "上传中..." : "上传文件到知识库"}</span>
            </label>
            {uploadState ? (
              <div className="upload-result">
                <div>{uploadState.filename}</div>
                <div>{uploadState.chunksInserted} chunks</div>
              </div>
            ) : null}
          </div>
        </div>

        {error ? <div className="error-card">{error}</div> : null}
      </aside>

      <main className="chat-column">
        <header className="chat-header compact-header">
          <div className="chat-header-row">
            <button className="mobile-toggle" type="button" onClick={() => setShowSidebar(true)}>
              菜单
            </button>
            <h1>Chat</h1>
            <button className="mobile-toggle" type="button" onClick={() => setShowTracePanel(true)}>
              运行
            </button>
          </div>
        </header>

        <section className="messages" ref={messagesRef} onScroll={updateAutoScrollLock}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>开始一个新对话</h2>
              <p>{userId ? "已进入该 USERID 的会话空间，直接提问或切换历史会话。" : "先输入 USERID 并点击确定，再开始聊天。"}</p>
            </div>
          ) : null}

          {messages.map((message) => (
            <article
              key={message.id}
              className={`message-row ${message.role}`}
              onClick={() => message.role === "assistant" && setActiveTraceMessageId(message.id)}
            >
              {(() => {
                const parsedPae = message.role === "assistant" ? parsePaeTrace(message.traces, message.content || "") : null;
                const fullThought = message.role === "assistant" ? message.thought || "" : "";
                const answer = message.content || "";
                const showCollapsedThought = Boolean(fullThought && answer);
                const thoughtExpanded = expandedThoughtIds.includes(message.id) || !showCollapsedThought;
                const bodyContent = answer || (fullThought ? "" : message.content || "处理中...");

                return (
                  <>
                    <div className="avatar">{message.role === "user" ? "你" : "AI"}</div>
                    <div className={`message-card ${message.role}`}>
                      <div className="message-meta">
                        <span>{message.role === "user" ? "You" : "Assistant"}</span>
                        {message.role === "assistant" ? (
                          <span className="trace-count">{message.traces.length} trace</span>
                        ) : null}
                      </div>
                      {message.role === "assistant" && parsedPae?.active ? (
                        <div className="pae-badge">PAE</div>
                      ) : null}
                      {message.role === "assistant" && fullThought ? (
                        <div className="thought-block">
                          <button
                            type="button"
                            className="thought-toggle"
                            onClick={() => setThoughtExpanded(message.id, !thoughtExpanded)}
                          >
                            {thoughtExpanded ? "▾ 思考过程（折叠）" : "▸ 思考过程（展开）"}
                          </button>
                          {thoughtExpanded ? (
                            <div className="thought-body-wrap">
                              <div
                                className="thought-body"
                                dangerouslySetInnerHTML={renderMarkdown(fullThought)}
                              />
                              <button
                                type="button"
                                className="thought-toggle thought-toggle-bottom"
                                onClick={() => setThoughtExpanded(message.id, false)}
                              >
                                ▴ 收起思考过程
                              </button>
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                      <div
                        className="message-body"
                        dangerouslySetInnerHTML={renderMarkdown(bodyContent)}
                      />
                    </div>
                  </>
                );
              })()}
            </article>
          ))}
          <div ref={bottomRef} />
        </section>

        <footer className="composer">
          <div className="composer-shell">
            <div className="prompt-presets">
              <span className="prompt-presets-label">体验示例</span>
              <div className="prompt-preset-list">
                {PROMPT_PRESETS.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    className="prompt-preset-chip"
                    onClick={() => applyPromptPreset(preset.prompt)}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
            <textarea
              ref={textareaRef}
              rows={1}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={userId.trim() ? "Type a message..." : "请先输入 USERID 并点击确定"}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void handleSend();
                }
              }}
            />
            <div className="composer-inner-toolbar">
              <label className="inline-select">
                  <span className="toolbar-label">模型</span>
                <select value={modelChoice} onChange={(e) => setModelChoice(e.target.value)}>
                  <option value="local_qwen">本地 Qwen 3.5 9B</option>
                  <option value="deepseek">DeepSeek Reasoner</option>
                  <option value="deepseek_chat">DeepSeek Chat</option>
                  <option value="minimax">MiniMax M2.7</option>
                  <option value="mimo">Xiaomi MiMo V2 Flash</option>
                  <option value="mimo_pro">Xiaomi MiMo V2 Pro</option>
                </select>
              </label>
              <button
                className={`toggle-button ${forcePlan ? "active" : ""}`}
                type="button"
                onClick={() => setForcePlan((value) => !value)}
              >
                <span className="toggle-full">强制 Plan-and-Execute 模式</span>
                <span className="toggle-short">强制 PAE</span>
              </button>
              <button
                className="primary-button"
                onClick={() => void handleSend()}
                disabled={!query.trim() || isSending}
              >
                {isSending ? "发送中" : "发送"}
              </button>
            </div>
          </div>
        </footer>
      </main>

      <aside className={`trace-panel ${showTracePanel ? "open" : ""}`}>
        <div className="trace-header run-header">
          <div>
            <h2>Agent Run</h2>
            {activeTraceMessage?.traces.length ? (
              <p>{activePaeTrace.active ? "Running" : "Trace Stream"}</p>
            ) : null}
          </div>
          <button className="drawer-close" type="button" onClick={() => setShowTracePanel(false)}>
            关闭
          </button>
        </div>
        <div className="trace-guide">
          <p>采用 ReAct 主循环，Auto 模式下自动判断进入 Plan-and-Execute 模式，也可手动强制开启。</p>
          <p>发送消息后，这里会流式显示推理、工具调用和 PAE 过程。</p>
        </div>
        <div className="trace-list">
          {activeTraceMessage?.traces.length ? (
            activePaeTrace.active ? (
              <div className="run-panel">
                <section className="run-section">
                  <div className="run-section-title">Execution Stages</div>
                  <div className="run-stage-list">
                    <div className={`run-stage ${activePaeTrace.entered ? "done" : ""}`}>进入 PAE</div>
                    <div className={`run-stage ${activePaeTrace.planning ? "done" : ""}`}>生成计划</div>
                    <div className={`run-stage ${activePaeTrace.executionSteps.length ? "done" : ""}`}>执行步骤</div>
                    <div className={`run-stage ${activePaeTrace.reflecting ? "done" : ""}`}>反思修正</div>
                    <div className={`run-stage ${activePaeTrace.generating ? "done" : ""}`}>生成答案</div>
                    <div className={`run-stage ${activePaeTrace.completed ? "done" : ""}`}>返回主循环</div>
                  </div>
                </section>

                {combinedPaeSteps.length ? (
                  <section className="run-section">
                    <div className="run-section-title">Plan & Execution</div>
                    <div className="run-step-list">
                      {combinedPaeSteps.map((step, index) => (
                        <div
                          key={`plan-${step.stepId}`}
                          className={`run-step-card ${activePaeTrace.currentStepId === step.stepId ? "active" : ""}`}
                        >
                          <div className="run-step-index">Step {index + 1}</div>
                          <div className="run-step-main">
                            <div className="run-step-top">
                              <strong>{step.goal}</strong>
                              <span>{activePaeTrace.currentStepId === step.stepId ? "running" : step.status}</span>
                            </div>
                            <div className="run-capability-chip">{step.capability}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                ) : null}

                {activePaeTrace.activatedSkills.length || activePaeTrace.allowedTools.length ? (
                  <section className="run-section">
                    <div className="run-section-title">Skills</div>
                    {activePaeTrace.activatedSkills.length ? (
                      <div className="run-capability-list">
                        {activePaeTrace.activatedSkills.map((skill) => (
                          <span key={skill} className="run-capability-chip">
                            {skill}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    {activePaeTrace.allowedTools.length ? (
                      <div className="run-submeta">
                        允许工具：{activePaeTrace.allowedTools.join("、")}
                      </div>
                    ) : null}
                  </section>
                ) : null}

                {activePaeTrace.mcpToolCalls.length ? (
                  <section className="run-section">
                    <div className="run-section-title">MCP Tools</div>
                    <div className="run-step-list">
                      {activePaeTrace.mcpToolCalls.map((toolName, index) => (
                        <div key={`${toolName}-${index}`} className="run-step-card">
                          <div className="run-step-main">
                            <div className="run-step-top">
                              <strong>{toolName}</strong>
                              <span>called</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                ) : null}

                {activePaeTrace.reflectionSteps.length ? (
                  <section className="run-section">
                    <div className="run-section-title">Reflection</div>
                    <div className="run-step-list">
                      {activePaeTrace.reflectionSteps.map((step, index) => (
                        <div key={`reflection-${step.stepId}-${index}`} className="run-step-card reflection">
                          <div className="run-step-main">
                            <div className="run-step-top">
                              <strong>{step.stepId}</strong>
                              <span>{step.status}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                ) : null}
              </div>
            ) : (
              activeTraceMessage.traces.map((trace, index) => (
                <pre key={`${activeTraceMessage.id}-${index}`} className="trace-item">
                  {trace}
                </pre>
              ))
            )
          ) : (
            <div className="trace-empty">尚无运行记录。</div>
          )}
        </div>
      </aside>
      {(showSidebar || showTracePanel) ? (
        <button
          className="drawer-overlay"
          type="button"
          aria-label="关闭抽屉"
          onClick={() => {
            setShowSidebar(false);
            setShowTracePanel(false);
          }}
        />
      ) : null}
      {toastMessage ? <div className="toast-message">{toastMessage}</div> : null}
    </div>
  );
}
