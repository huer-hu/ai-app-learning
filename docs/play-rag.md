# 玩转 RAG：解锁大模型的进阶玩法

## 一、RAG：大模型时代的得力助手

> 代码示例在: [play-rag](../example/play-rag/index.ts)

### 1.1 大模型的短板

大模型虽强大，但存在两大明显不足：一是缺乏实时性和知识更新能力，其训练数据截至某一时间点，之后的新信息无法获取，且频繁更新模型成本高昂；二是知识覆盖的局限性，无法涵盖所有领域知识，尤其是私有数据，这就导致在回答问题时容易出现 "幻觉"，答案缺乏可信度。

### 1.2 RAG 的独特价值

RAG（检索增强生成）通过为大模型外挂知识库，有效解决了上述问题。它能在回答问题时参考外挂知识库中的知识，使大模型生成的答案更精确、贴合上下文，同时减少误导性信息的产生。

## 二、RAG 的核心原理揭秘

### 2.1 RAG 的工作流程

RAG 的工作流程可以拆解为三个紧密相连的关键步骤，每个步骤都承载着独特的使命，共同为生成高质量的回答贡献力量 。

**检索**：当用户输入一个查询时，系统首先会把这个查询转化为向量形式，这个过程就像是给查询生成一个独特的 "数字指纹"。随后，这个向量会被送入向量数据库，与数据库中已有的知识向量进行比对。通过计算向量之间的相似度，系统能够找出最匹配的前 K 个数据，这些数据就是与用户查询最相关的知识片段。

**增强**：检索到相关知识后，系统会将用户查询和这些知识整合到预设的提示词模板中。提示词模板就像是一个精心设计的框架，它能够引导大语言模型更好地理解问题和相关知识，从而为生成准确的回答做好充分准备。

**生成**：经过增强的提示词被输入到大语言模型中，大语言模型会基于这些信息进行深度学习和推理，最终生成用户所需的输出。这个输出可能是一个回答、一段文本或者其他形式的内容，其质量和准确性直接取决于前面两个步骤的执行效果。

### 2.2 核心技术剖析

**向量检索**：向量检索是 RAG 的核心技术之一，它依赖于强大的嵌入模型，如 Sentence Transformers，来将文本转化为向量。这些向量能够高效地捕捉文本的语义信息，使得文本之间的相似度比较更加精准。在实际应用中，我们通常会使用余弦相似度等算法来计算向量之间的相似度，以此来衡量文本之间的相关性。例如，假设我们有两个文本片段 A 和 B，通过嵌入模型将它们转化为向量 a 和 b，然后计算 a 和 b 的余弦相似度，如果相似度越高，就说明 A 和 B 在语义上越相近。

**混合检索策略**：为了进一步提升检索的精准度和全面性，RAG 常常采用混合检索策略，将向量检索和关键词检索有机结合。向量检索擅长捕捉语义相似性，能够找到那些在语义上与查询相关但关键词可能不同的文档；而关键词检索则可以快速定位包含特定关键词的文档。通过这种方式，RAG 可以更全面地检索到与用户查询相关的信息，避免遗漏重要内容。

## 三、RAG 系统的实现指南

### 3.1 环境搭建

在开始搭建 RAG 系统之前，确保你的开发环境满足以下要求：

在上一节中，我们已经使用了 Bunjs 完成一个简单的问答逻辑，这里我们继续使用 Bunjs 完成一个完整的 RAG 系统。

**创建项目目录**：在你喜欢的位置创建一个新的项目目录，例如`play-rag`，并进入该目录：

```bash
mkdir play-rag

cd play-rag
```

**初始化项目**：运行`bun init -y`命令，初始化一个新的 JavaScript 项目，并生成`package.json`文件。

**安装依赖包**：我们需要安装一些必要的依赖包，如 `sqlite-vec` 用于向量数据库操作，`openai` 用于调用 OpenAI 的大语言模型，`@xenova/transformers` 用于文本嵌入。运行以下命令进行安装：

```bash
bun add sqlite-vec openai @xenova/transformers
```

### 3.2 数据准备与索引构建

**数据收集**：首先，收集你想要用于 RAG 系统的文本数据。这些数据可以是来自各种来源，如文档、网页、数据库等。例如，我们收集了一些关于人工智能的文章作为示例数据。

```typescript
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
```

**数据清洗**：对收集到的数据进行清洗，去除噪声、重复数据和无关信息。这一步可以使用正则表达式、字符串操作等方法来实现。

```typescript
function cleanData(data: string) {
  // 在后面的段落中我们会介绍去重和降噪的方法
  // 这里假设我们进行了一些数据清洗操作
  return data.trim();
}
```

**文本分块**：将清洗后的数据分割成较小的文本块，以便于后续的处理和检索。分块的大小可以根据实际情况进行调整，一般建议在 100 - 500 个词之间。这里使用重叠分块策略：

```javascript
function splitTextIntoChunks(text: string, chunkSize = 100, overlapSize = 20) {
  // 进行基础的分割，保留一定语义的数据块
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
```

**生成向量嵌入**：使用嵌入模型将每个文本块转换为向量表示。这里我们使用`@xenova/transformers`库中的`AutoModel`来生成向量嵌入：

```typescript
import { AutoModel, AutoTokenizer } from "@xenova/transformers";

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
    // 确保维度是数据库定义的维度
    while (embedding.length < VECTOR_DIM) {
      embedding.push(0);
    }

    embeddings.push(embedding);
  }

  return embeddings;
}
```

**索引构建**：将生成的向量嵌入存储到向量数据库中，如 `sqlite-vec`。首先，初始化 SQLite 数据库，并创建一个新的表：

```typescript
import * as sqliteVec from "sqlite-vec";
import { Database } from "bun:sqlite";

// 如果是macos，需要设置sqlite路径
// 推荐使用brew安装sqlite
Database.setCustomSQLite("/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib");

const db = new Database(":memory:");
sqliteVec.load(db);

// 定义向量维度
const VECTOR_DIM = 768;
db.exec(`
  CREATE VIRTUAL TABLE documents USING vec0(
    embedding FLOAT[${VECTOR_DIM}],
    content TEXT
  );
`);
```

然后，将文本块和对应的向量嵌入添加到数据库中：

```typescript
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
```

### 3.3 检索与生成

**查询向量生成**：当用户输入一个查询时，将查询转换为向量表示，使用与生成索引时相同的嵌入模型：

```typescript
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
```

**向量检索**：使用查询向量在向量数据库中进行检索，找到最相关的文本块：

```typescript
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
```

**答案生成**：将检索到的文本块与用户查询组合，作为提示输入到大语言模型中，生成最终的答案。这里我们依然使用 DeepSeek 的模型：

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: "你的 DeepSeek API Key",
  baseURL: "https://api.deepseek.com/v1",
});

async function generateAnswer(retrievedDocs: RetrievedDocument[], query: string): Promise<string> {
  try {
    const context = retrievedDocs.map((doc) => doc.content).join("\n");
    const prompt = `查询: ${query}\n\n上下文: ${context}\n\n回答:`;

    const response = await openai.chat.completions.create({
      model: "deepseek-chat",
      messages: [{ role: "user", content: prompt }],
    });

    const messageContent = response.choices[0]?.message?.content;
    return messageContent !== null && messageContent !== undefined ? messageContent : "无回答内容";
  } catch (error) {
    console.error("生成回答时出错:", error);
    return "生成回答时发生错误";
  }
}
```

## 四、RAG 与传统生成模式的对比

为了更直观地理解 RAG 的优势，我们将其与传统的生成模式进行详细对比 。

| 特性     | 传统生成模式     | RAG 增强模式   |
| -------- | ---------------- | -------------- |
| 知识来源 | 依赖模型内部参数 | 结合外部知识库 |
| 准确性   | 易产生幻觉       | 基于事实生成   |
| 时效性   | 滞后于最新数据   | 可动态更新     |
| 可解释性 | 难以追溯来源     | 提供引用依据   |

### 4.1 知识来源的差异

传统的生成**模式完全依赖于**模型在训练阶段学到的知识，这些知识被编码在模型的参数中。一旦模型训练完成，其知识储备就固定下来，无法实时获取新的信息。而 RAG 模式则引入了外部知识库，通过向量检索等技术从知识库中获取与问题相关的最新知识，使得模型在生成回答时能够结合最新的信息，大大扩展了知识的来源。

### 4.2 准确性与幻觉问题

传统生成模式在回答问题时，由于缺乏实时的知识支持，容易产生 "幻觉"，即生成看似合理但实际上错误的回答。例如，当询问 "2024 年的最新科研成果有哪些" 时，传统模型可能会根据其训练数据中的历史信息进行猜测，给出过时或错误的答案。而 RAG 模式通过检索相关的最新文献和资料，基于真实的信息生成回答，能够有效减少幻觉的产生，提高回答的准确性。

### 4.3 时效性的对比

在信息快速更新的时代，时效性至关重要。传统生成模式由于无法及时更新知识，对于涉及最新事件、数据和研究成果的问题，往往无法给出准确的回答。RAG 模式则可以通过定期更新知识库，或者实时连接到最新的数据源，确保在生成回答时能够使用最新的信息，从而提供更具时效性的答案。

### 4.4 可解释性的提升

传统生成模式生成的回答往往难以追溯其知识来源，这在一些对答案准确性和可靠性要求较高的场景中是一个明显的缺陷。而 RAG 模式在生成回答时，会同时提供相关的引用依据，即检索到的文本片段，这使得用户可以清楚地了解答案的来源，增强了回答的可解释性和可信度。

## 五、扩展内容：RAG 的优化策略

为了进一步提升 RAG 系统的性能和可靠性，我们可以采用一系列优化策略，从检索方式、数据处理到输出控制，全面提升系统的表现。

### 5.1 混合检索与多级过滤

**混合检索**：在 RAG 系统中，将向量检索和关键词检索相结合，可以充分发挥两者的优势。向量检索擅长捕捉语义相似性，而关键词检索则能快速定位包含特定关键词的文档。例如，当用户查询 "iPhone 15" 时，系统可以优先召回包含 "iPhone 15" 这个关键词的文档，然后再使用向量检索对这些文档进行进一步筛选，以确保检索结果的准确性和相关性。在 Bunjs 中，可以使用以下代码实现简单的混合检索：

```typescript
async function hybridSearch(db, query, queryVector, topK = 5) {
  // 先通过关键词过滤
  const keywordResults = db
    .prepare(
      `
    SELECT 
      rowid,
      content
    FROM documents
    WHERE content LIKE ?
    LIMIT ?
  `
    )
    .all(`%${query}%`, topK * 2);

  // 再通过向量相似度过滤
  const vectorResults = db
    .prepare(
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

  // 合并结果（去重）
  const seenIds = new Set();
  const combinedResults = [];

  // 优先添加向量检索结果
  for (const result of vectorResults) {
    seenIds.add(result.rowid);
    combinedResults.push({
      id: result.rowid,
      content: result.content,
      distance: result.distance,
      source: "vector",
    });
  }

  // 添加未出现在向量检索结果中的关键词检索结果
  for (const result of keywordResults) {
    if (!seenIds.has(result.rowid) && combinedResults.length < topK) {
      seenIds.add(result.rowid);
      combinedResults.push({
        id: result.rowid,
        content: result.content,
        distance: 999, // 设一个较大的距离值
        source: "keyword",
      });
    }
  }

  return combinedResults.slice(0, topK);
}
```

**多级过滤**：为了提高检索结果的质量，可以采用多级过滤策略。

**粗过滤**：通过设置余弦相似度阈值，筛选出候选文档。例如，只保留余弦相似度大于 0.8 的文档，这样可以快速缩小检索范围，减少后续处理的工作量，可以使用 `@langchain/community/vectorstores/hnswlib` 来实现。

**重排序**：使用 BM25 或机器学习模型对候选文档进行重新排序，以提高相关性最高的文档排在前面。在 Bunjs 中，可以使用`@langchain/community/retrievers/bm25`库来实现 BM25 重排序：

**精过滤**：基于元数据（如文档类型、时效性）进一步筛选文档。例如，只保留最近一个月内更新的文档，或者只保留特定类型的文档（如 PDF 文档），以确保检索结果的时效性和相关性。

### 5.2 智能分块与数据清洗

**分块策略**：在将文本数据存储到向量数据库之前，需要将其分割成较小的块，以便于检索和处理。采用递归字符分割法是一种有效的分块策略，设置块大小为 200，重叠度为 50，可以在保证语义完整性的同时，提高检索效率。在 Bunjs 中，可以使用`@langchain/textsplitters`库来实现递归字符分割：

**数据清洗**：数据清洗是确保 RAG 系统性能的关键步骤，主要包括去重和降噪两个方面。

**去重**：利用 SimHash 算法检测重复内容，去除重复的文本块，以减少数据冗余，提高检索效率。在 Bunjs 中，可以使用`simhash-js`库来实现 SimHash 去重：

**降噪**：过滤低质量文本，如广告、乱码等，以提高数据的质量。可以使用正则表达式、文本分类模型等方法来实现降噪。例如，使用正则表达式去除文本中的 HTML 标签和 URL 链接：

## 六、总结

RAG 技术作为大模型能力扩展的有力工具，通过将检索与生成紧密结合，有效弥补了大模型在知识时效性、准确性和专业性方面的不足。在本文中，我们深入探讨了 RAG 的核心原理、实现方法和优化策略，为开发者提供了一套完整的构建高效 RAG 系统的指南。

随着大模型技术的迅速发展，RAG 系统也在不断进化。未来，我们可以期待以下几个方向的发展：

1. **多模态 RAG**：将文本、图像、音频等多种模态的数据整合到一个统一的检索框架中，使 RAG 系统能够理解和处理更丰富的信息形式。

2. **自适应检索**：根据查询的类型和上下文动态调整检索策略，实现更智能、更精准的知识获取。

3. **知识图谱增强**：将结构化的知识图谱与非结构化文本相结合，提供更全面、更深入的知识支持。

4. **个性化 RAG**：根据用户的历史交互和偏好，为不同用户提供定制化的检索结果和生成内容。

通过不断探索和创新，RAG 技术将继续发挥其独特价值，为大模型赋予更强大的能力，为用户带来更优质的智能体验。在实践中，我们也应当持续关注 RAG 的最新进展，不断优化系统性能，使其在实际应用中发挥最大效用。

无论是企业级应用还是个人项目，掌握 RAG 技术都将成为构建下一代智能系统的关键能力。让我们一同探索，共同推动 RAG 技术的发展与应用，解锁大模型的无限可能。
