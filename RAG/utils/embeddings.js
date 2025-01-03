const axios = require("axios");

// Replace this with your embedding model or API
const getEmbeddings = async (text) => {
  try {
    const response = await axios.post("http://localhost:8000/embeddings", { text });
    return response.data.embeddings;
  } catch (err) {
    console.error("Error generating embeddings:", err);
    throw err;
  }
};

module.exports = getEmbeddings;
