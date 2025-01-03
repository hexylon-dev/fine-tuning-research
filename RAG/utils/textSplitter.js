const splitText = (text, chunkSize = 500) => {
  const paragraphs = text.split("\n").filter((p) => p.trim() !== "");
  return paragraphs;
};

module.exports = splitText;
