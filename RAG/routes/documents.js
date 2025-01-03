const express = require("express");
const multer = require("multer");
const path = require("path");
const pdfToText = require("../utils/pdfToText");
const splitText = require("../utils/textSplitter");
const getEmbeddings = require("../utils/embeddings");
const axios = require("axios");
const router = express.Router();
const { DefaultEmbeddingFunction } = require("chromadb");
const client = require("./connection");
const defaultEF = new DefaultEmbeddingFunction();

let collection;
// Initialize collection
(async () => {
  try {
    collection = await client.getCollection({
      name: "my_collection"
    });
  } catch (err) {
    console.error("Error initializing collection:", err);
  }
})();

// Configure multer for file upload
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/"); // Make sure this directory exists
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  fileFilter: function (req, file, cb) {
    // Accept PDF files only
    if (path.extname(file.originalname).toLowerCase() === ".pdf") {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are allowed"));
    }
  },
});

router.post("/add", upload.single("file"), async (req, res) => {
  let collection = await client.getOrCreateCollection({
    name: "my_collection",
  });

  try {
    let chunks;

    if (req.file) {
      // If a file was uploaded, process the PDF
      chunks = await pdfToText(req.file.path);
    } else if (req.body.text) {
      // If text was provided directly
      chunks = req.body.text;
    } else {
      return res
        .status(400)
        .json({ error: "Either file or text must be provided" });
    }

    // const chunks = splitText(chunks);
    const embeddings = await Promise.all(
      chunks.map(async (text) => defaultEF.generate([text]))
    );

    const ids = [];
    for (let i=0; i<chunks.length; i++) ids.push(crypto.randomUUID())

      console.log(embeddings)

    await collection.add({
      ids,
      documents: chunks,
      embeddings: embeddings,
    });

    // Clean up the uploaded file
    if (req.file) {
      const fs = require("fs").promises;
      await fs.unlink(req.file.path);
    }

    res.json({ message: "Document added successfully" });
  } catch (err) {
    // Clean up file on error
    if (req.file) {
      const fs = require("fs").promises;
      try {
        await fs.unlink(req.file.path);
      } catch (unlinkErr) {
        console.error("Error deleting file:", unlinkErr);
      }
    }

    console.error(err);
    res.status(500).json({
      error: "Failed to add document",
      details: err.message,
    });
  }
});

router.delete("/deleteAll", async (req, res) => {
  try {
    if (!collection) {
      return res.status(500).json({
        error: "Collection not initialized",
        status: "error"
      });
    } 

    await client.deleteCollection({
      name: "my_collection"
    });
    
    collection = await client.createCollection({
      name: "my_collection"
    });

    res.json({
      message: "Successfully deleted all documents and recreated collection",
      status: "success"
    });

  } catch (err) {
    console.error("Error deleting documents:", err);
    
    res.status(500).json({
      error: "Failed to delete documents",
      details: err.message,
      status: "error"
    });
  }
});

// Delete specific documents by IDs
router.delete("/delete", async (req, res) => {
  try {
    const { ids } = req.body;

    if (!collection) {
      return res.status(500).json({
        error: "Collection not initialized",
        status: "error"
      });
    }

    if (!Array.isArray(ids) || ids.length === 0) {
      return res.status(400).json({
        error: "Invalid input: ids must be a non-empty array",
        status: "error"
      });
    }

    // Delete in batches to avoid memory issues
    const BATCH_SIZE = 100;
    let deletedCount = 0;

    for (let i = 0; i < ids.length; i += BATCH_SIZE) {
      const batch = ids.slice(i, i + BATCH_SIZE);
      await collection.delete({
        ids: batch
      });
      deletedCount += batch.length;
    }

    res.json({
      message: "Successfully deleted specified documents",
      deletedCount,
      status: "success"
    });

  } catch (err) {
    console.error("Error deleting documents:", err);
    
    res.status(500).json({
      error: "Failed to delete documents",
      details: err.message,
      status: "error"
    });
  }
});

module.exports = router;
