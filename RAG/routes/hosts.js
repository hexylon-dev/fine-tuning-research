const sqlite3 = require('sqlite3').verbose();
const db = new sqlite3.Database('hosts.db');
const express = require("express");
const router = express.Router();

db.run("CREATE TABLE IF NOT EXISTS hosts (hostname TEXT PRIMARY KEY, domain TEXT NOT NULL)");

function getDomain(hostname){
    db.get(`SELECT * FROM hosts where hostname = ${hostname}`, (err, row)=>{
        return row.domain;
    })

    return false;
}

function setDomain(hostname, domain){
    db.run("INSERT OR REPLACE INTO hosts (hostname, person) VALUES (?, ?)", [hostname, domain]);
}

router.get("/", (req, res)=>{
    setDomain(req.hostname, req.domain)
    res.send({"success": true})
});

module.exports = {getDomain, setDomain, hostRouter: router}