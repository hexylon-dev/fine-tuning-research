const express = require("express");
const bodyParser = require("body-parser");
const documentsRouter = require("./routes/documents");
const queryRouter = require("./routes/query");
const {hostRouter} = require("./routes/hosts");

const app = express();
const PORT = 6000;
app.use(bodyParser.json());
app.use("/documents", documentsRouter);
app.use("/query", queryRouter);
app.use("/hosts", hostRouter);

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
