// 从 bun 导入 serve 函数，用于创建 HTTP 服务器
import { serve } from "bun";
// 从 @langchain/deepseek 导入 ChatDeepSeek 类，用于与 DeepSeek AI 模型交互
import { ChatDeepSeek } from "@langchain/deepseek";
// 从 fs 模块导入 readFileSync 函数，用于同步读取文件
import { readFileSync } from "fs";

// 创建 ChatDeepSeek 实例，配置模型名称和 API 密钥
const model = new ChatDeepSeek({
  model: "deepseek-chat", // 指定使用的模型名称
  // 提供 API 密钥进行身份验证，
  // 注册 DeepSeek 账号后，从 https://platform.deepseek.com/api_keys 中获取
  // 获取后替换掉下面的 API 密钥
  apiKey: "你的 API 密钥",
});

// 同步读取 index.html 文件内容，并以 UTF-8 编码方式存储到 indexHtml 变量
const indexHtml = readFileSync("index.html", "utf8");

// 创建并启动 HTTP 服务器，监听 3003 端口
serve({
  port: 3003, // 设置服务器监听的端口号
  async fetch(req) {
    // 定义处理请求的异步函数
    // 解析请求 URL
    const url = new URL(req.url);

    // 处理 GET 请求且路径为根路径 "/"
    if (req.method === "GET" && url.pathname === "/") {
      // 返回 HTML 文件内容作为响应
      return new Response(indexHtml, {
        headers: { "Content-Type": "text/html" }, // 设置响应头，指定内容类型为 HTML
      });
    }

    // 处理 POST 请求且路径为 "/ask"
    if (req.method === "POST" && url.pathname === "/ask") {
      // 从请求体中解析 JSON 数据，获取 question 字段
      const { question } = await req.json();
      // 调用 AI 模型处理用户问题，获取回答
      const answer = await model.invoke([{ role: "user", content: question }]);
      // 返回 JSON 格式的响应，包含 AI 模型的回答
      return new Response(JSON.stringify({ answer: answer.content }), {
        headers: { "Content-Type": "application/json" }, // 设置响应头，指定内容类型为 JSON
      });
    }

    // 处理所有其他未匹配的请求，返回 404 错误
    return new Response("未找到", { status: 404 });
  },
});
