import { AutoModel, AutoTokenizer } from "@xenova/transformers";
import * as sqliteVec from "sqlite-vec";
import { Database } from "bun:sqlite";
import OpenAI from "openai";

// 如果是macos，需要设置sqlite路径
// 推荐使用brew安装sqlite
Database.setCustomSQLite("/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib");

const db = new Database(":memory:");
sqliteVec.load(db);

// 定义向量维度
const VECTOR_DIM = 768;

// 创建向量表
db.exec(`
  CREATE VIRTUAL TABLE documents USING vec0(
    embedding FLOAT[${VECTOR_DIM}],
    content TEXT
  );
`);

// 示例文档
const documents = [
  "向量检索是一种在高维空间中查找最相似向量的技术。",
  "RAG（检索增强生成）通过检索外部知识来增强大型语言模型的能力。",
  "SQLite是一个轻量级的关系型数据库，被广泛应用于嵌入式系统。",
  "sqlite-vec是一个为SQLite添加向量搜索功能的扩展。",
  "语义搜索通过理解查询和文档的含义来提高搜索相关性。",
  "嵌入向量（Embedding）是将文本、图像等数据映射到高维向量空间的表示方法。",
  "向量数据库专门设计用于存储和检索向量数据，支持相似度搜索。",
  "余弦相似度是衡量两个向量方向相似性的指标，常用于文本相似度计算。",
  "大型语言模型（LLM）是基于深度学习的自然语言处理模型，能够理解和生成人类语言。",
  "知识图谱是一种结构化的知识表示方式，用于描述实体及其关系。",
];

function cleanData(data: string): string {
  // 只移除不必要的字符，保留更多原文信息
  return data.trim();
}

function splitTextIntoChunks(text: string, chunkSize = 100, overlapSize = 20): string[] {
  // 以下是原来的分割逻辑
  const sentences = text.split(/(?<=[.!?])\s+/);
  const chunks: string[] = [];
  let currentChunk: string[] = [];

  for (const sentence of sentences) {
    const sentenceLength = sentence.split(" ").length;
    if (currentChunk.reduce((acc, s) => acc + s.split(" ").length, 0) + sentenceLength > chunkSize) {
      chunks.push(currentChunk.join(" ").trim());
      currentChunk = currentChunk.slice(-overlapSize); // 保留重叠部分
    }
    currentChunk.push(sentence);
  }

  if (currentChunk.length) {
    chunks.push(currentChunk.join(" ").trim());
  }

  return chunks;
}

async function generateEmbeddings(chunks: string[]): Promise<number[][]> {
  // 使用与查询向量相同的模型，保证维度一致
  const model = await AutoModel.from_pretrained("Xenova/all-MiniLM-L6-v2");
  const tokenizer = await AutoTokenizer.from_pretrained("Xenova/all-MiniLM-L6-v2");

  const embeddings: number[][] = [];
  for (const chunk of chunks) {
    const encoded = await tokenizer(chunk);
    const output = await model(encoded, { pooling: "mean" });

    let embedding: any = [];
    const hiddenState = (output as any).last_hidden_state;
    const hiddenStateArray = hiddenState.tolist();
    embedding = hiddenStateArray[0][0];

    while (embedding.length < VECTOR_DIM) {
      embedding.push(0);
    }

    embeddings.push(embedding);
  }

  return embeddings;
}

const insertStmt = db.prepare("INSERT INTO documents(rowid, embedding, content) VALUES (?, vec_f32(?), ?)");

async function insertDocuments(chunks: string[], embeddings: number[][]): Promise<void> {
  const insertVectors = db.transaction((items) => {
    for (let i = 0; i < items.length; i++) {
      const [id, vector, content] = items[i];
      insertStmt.run(BigInt(id), new Float32Array(vector), content);
    }
  });

  // 准备要插入的数据
  const itemsToInsert = chunks.map((chunk, index) => [index + 1, embeddings[index], chunk]);
  insertVectors(itemsToInsert);
}

async function generateQueryEmbedding(query: string): Promise<number[]> {
  const model = await AutoModel.from_pretrained("Xenova/all-MiniLM-L6-v2");
  const tokenizer = await AutoTokenizer.from_pretrained("Xenova/all-MiniLM-L6-v2");

  const encoded = await tokenizer(query);
  const output = await model(encoded, { pooling: "mean" });

  // 提取嵌入向量
  let embedding = (output as any).last_hidden_state.tolist()[0][0];

  while (embedding.length < VECTOR_DIM) {
    embedding.push(0);
  }

  return embedding;
}

interface RetrievedDocument {
  id: number;
  content: string;
  distance: number;
}

async function retrieveDocuments(database: Database, queryVector: number[], topK = 5): Promise<RetrievedDocument[]> {
  const results = database
    .query(
      `
      SELECT 
        rowid,
        content,
        distance
      FROM documents
      WHERE embedding MATCH ?
      ORDER BY distance
      LIMIT ?
    `
    )
    .all(JSON.stringify(queryVector), topK);

  return results.map((row: any) => ({
    id: row.rowid,
    content: row.content,
    distance: row.distance,
  }));
}

// 初始化OpenAI客户端
const openai = new OpenAI({
  apiKey: "",
  baseURL: "https://api.deepseek.com/v1",
});

async function generateAnswer(retrievedDocs: RetrievedDocument[], query: string): Promise<string> {
  try {
    const context = retrievedDocs.map((doc) => doc.content).join("\n");
    const prompt = `查询: ${query}\n上下文: ${context}\n回答:`;

    const response = await openai.chat.completions.create({
      model: "deepseek-chat",
      messages: [{ role: "user", content: prompt }],
    });

    if (!response || !response.choices || response.choices.length === 0) {
      return "无法获取回答";
    }

    const messageContent = response.choices[0]?.message?.content;
    return messageContent !== null && messageContent !== undefined ? messageContent : "无回答内容";
  } catch (error) {
    console.error("生成回答时出错:", error);
    return "生成回答时发生错误";
  }
}

// 示例使用流程
async function main() {
  try {
    // 处理文档
    const cleanedDocs = documents.map(cleanData);
    const chunks = cleanedDocs.flatMap((doc) => splitTextIntoChunks(doc));

    // 生成嵌入
    const embeddings = await generateEmbeddings(chunks);

    // 存储文档
    await insertDocuments(chunks, embeddings);

    // 示例查询
    const query = "什么是RAG技术?";
    const queryEmbedding = await generateQueryEmbedding(query);

    // 检索相关文档
    const retrievedDocs = await retrieveDocuments(db, queryEmbedding);
    console.log("🚀 ~ main ~ retrievedDocs:", retrievedDocs);
    // retrievedDocs 输出：
    // [
    //   {
    //     id: 2,
    //     content: "RAG（检索增强生成）通过检索外部知识来增强大型语言模型的能力。",
    //     distance: 3.775055170059204,
    //   },
    //   {
    //     id: 7,
    //     content: "向量数据库专门设计用于存储和检索向量数据，支持相似度搜索。",
    //     distance: 5.395895004272461,
    //   },
    //   {
    //     id: 10,
    //     content: "知识图谱是一种结构化的知识表示方式，用于描述实体及其关系。",
    //     distance: 5.453629016876221,
    //   },
    //   {
    //     id: 9,
    //     content: "大型语言模型（LLM）是基于深度学习的自然语言处理模型，能够理解和生成人类语言。",
    //     distance: 5.732015132904053,
    //   },
    //   {
    //     id: 6,
    //     content: "嵌入向量（Embedding）是将文本、图像等数据映射到高维向量空间的表示方法。",
    //     distance: 5.815735340118408,
    //   },
    // ];

    // 生成回答
    const answer = await generateAnswer(retrievedDocs, query);
    console.log("查询:", query);
    console.log("回答:", answer);
  } catch (error) {
    console.error("执行过程中出错:", error);
  }
}

main();
