const pdf = require("pdf-parse");
const fs = require("fs");

const pdfToPageWiseText = async (filePath) => {
  const dataBuffer = fs.readFileSync(filePath);
  const data = await pdf(dataBuffer);

  const pages = data.text.split("\n\n").filter((page) => page.trim() !== "");
  
  return pages.map((pageText, pageIndex) => pageText.trim());
};

module.exports = pdfToPageWiseText;
