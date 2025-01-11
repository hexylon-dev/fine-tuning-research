const express = require("express");
const client = require("./connection");
const router = express.Router();
const { DefaultEmbeddingFunction } = require("chromadb");
const {getDomain} = require("./hosts")
const defaultEF = new DefaultEmbeddingFunction();

router.post("/", async (req, res) => {
  try {
    let { query, limit = 10, domain } = req.body;
    if(domain === undefined){
      try{
        domain = getDomain(req.hostname)
        if(domain === false){
          return res.status(404).json({ 
            error: `Collection for ${req.hostname} not found` 
          });
        }
      }catch(err){
        console.error('Failed to get collection:', err);
        return res.status(404).json({ 
          error: `Collection for ${req.hostname} not found` 
        });
      }
    }
    let collection;
    try {
      collection = await client.getCollection({
        name: domain,
      });
    } catch (error) {
      console.error('Failed to get collection:', error);
      return res.status(404).json({ 
        error: `Collection '${domain}' not found` 
      });
    }

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
