import { defineConfig } from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "跨越前端边界.md",
  description: "专为前端工程师量身打造，通过 14 + 篇系统化课程，带你从 Web 开发平滑过渡到 AI 大模型应用开发",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [{ text: "Home", link: "/" }],

    sidebar: [
      {
        text: "Examples",
        items: [{ text: "Runtime API Examples", link: "/api-examples" }],
      },
    ],

    socialLinks: [{ icon: "github", link: "https://github.com/vuejs/vitepress" }],
  },
});
