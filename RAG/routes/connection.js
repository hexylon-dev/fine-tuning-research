const { ChromaClient } = require("chromadb");
const client = new ChromaClient({
    path: "http://localhost:8000"
});
module.exports = client