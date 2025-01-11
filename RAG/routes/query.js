const express = require("express");
const client = require("./connection");
const router = express.Router();
const { DefaultEmbeddingFunction } = require("chromadb");
const defaultEF = new DefaultEmbeddingFunction();

router.post("/", async (req, res) => {
  try {
    let { query, limit = 10 } = req.body;

    let collection = await client.getCollection({
      name: "lom",
    });

    const queryEmbedding = await defaultEF.generate(query);

    const response = await collection.query({
      queryEmbeddings: queryEmbedding,
      nResults: limit,
    });

    const average = response.distances[0].reduce((acc, item) => item + acc, 0) / response.distances[0].length;
    const relevantChunks = [];

    response.distances[0].forEach((item, index) => {
      (item < average) ? relevantChunks.push(response.documents[index]) : null;
    });

    res.json({ context: relevantChunks.join(".  "), response });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to process query" });
  }
});

module.exports = router;
