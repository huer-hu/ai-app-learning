import { OpenAI } from "openai";

const client = new OpenAI({
  apiKey: "",
  baseURL: "https://api.deepseek.com/v1",
});

// 假设定义天气查询函数
async function getWeather(city: string, date: string) {
  return `${date}${city}会下雨`;
}

async function handleWeatherQuery(userQuery: string) {
  const response = await client.chat.completions.create({
    model: "deepseek-chat",
    messages: [{ role: "user", content: userQuery }],
    tools: [
      {
        type: "function",
        function: {
          name: "getWeather",
          description: "获取指定城市的天气预报",
          parameters: {
            type: "object",
            properties: {
              city: {
                type: "string",
                description: "城市名称，如'北京'、'上海'",
              },
              date: {
                type: "string",
                description: "查询日期，格式为YYYY-MM-DD",
              },
            },
            required: ["city"],
          },
        },
      },
    ],
    tool_choice: "auto",
  });

  const responseMessage = response.choices[0]?.message;
  if (responseMessage?.tool_calls && responseMessage.tool_calls.length > 0) {
    for (const toolCall of responseMessage.tool_calls) {
      if (toolCall.type === "function") {
        const functionName = toolCall.function.name;
        const functionArgs = JSON.parse(toolCall.function.arguments);

        console.log(`正在调用函数: ${functionName}`);
        console.log(`参数: ${JSON.stringify(functionArgs)}`);

        let functionResponse;
        if (functionName === "getWeather") {
          functionResponse = await getWeather(functionArgs.city, functionArgs.date);
        }

        const messages: Array<OpenAI.Chat.ChatCompletionMessageParam> = [
          { role: "user", content: userQuery },
          responseMessage as OpenAI.Chat.ChatCompletionMessageParam,
          {
            role: "tool",
            tool_call_id: toolCall.id,
            content: JSON.stringify(functionResponse),
          },
        ];

        const secondResponse = await client.chat.completions.create({
          model: "deepseek-chat",
          messages: messages,
        });

        return secondResponse.choices[0]?.message?.content;
      }
    }
  } else {
    return responseMessage?.content;
  }
}

const query = "今天是2025年04月06日，明天上海会下雨吗？";
console.log("用户问题:", query);

handleWeatherQuery(query).then((response) => {
  console.log("AI回复:", response);
  //output: 根据天气预报，2025年4月7日上海会下雨。建议您出门时携带雨具，注意出行安全
});
