const { ChromaClient } = require("chromadb");

const client = new ChromaClient({
  path: "http://192.168.1.22:8000",
});

module.exports = client;