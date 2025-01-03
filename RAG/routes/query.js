const express = require("express");
const getEmbeddings = require("../utils/embeddings");
const axios = require("axios");
const client = require("./connection");
const router = express.Router();
const { DefaultEmbeddingFunction } = require("chromadb");
const defaultEF = new DefaultEmbeddingFunction();

router.post("/", async (req, res) => {
  try {
    const { query } = req.body;
    
    let collection = await client.getOrCreateCollection({
      name: "my_collection",
    });

    const queryEmbedding = await defaultEF.generate([query]);

    const response = await collection.peek({
      // queryEmbeddings: queryEmbedding,
      limit: 10
      // nResults: 10,
  })

    // const relevantChunks = response.data.matches.map((match) => match.document);

    res.json({ query, response });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to process query" });
  }
});

module.exports = router;
