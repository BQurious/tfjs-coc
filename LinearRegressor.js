"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const csv = require("csv-parse");
require("@tensorflow/tfjs-node");
const t = require("runtypes");
const unitSpecFile = './unitSpecs.csv';
const structSpecFile = './structSpecs.csv';
const UnitSpecDef = t.Record({
    name: t.String,
    pop: t.Number,
    hp: t.Number,
    spd: t.Number,
    dps: t.Number,
    range: t.Number,
    isFlying: t.Boolean,
    targets: t.Array(t.Number)
});
var unitSpecArr;
function onUnitSpecReadComplete(data) {
    unitSpecArr = t.Array(UnitSpecDef).check(data);
    onReadComplete();
}
;
const StructSpecDef = t.Record({
    name: t.String,
    pop: t.Number,
    hp: t.Number,
    spd: t.Number,
    dps: t.Number,
    range: t.Number,
    isFlying: t.Boolean,
    targets: t.Array(t.Number)
});
var structSpecArr;
function onStructSpecReadComplete(data) {
    structSpecArr = t.Array(StructSpecDef).check(data);
    onReadComplete();
}
;
function onReadComplete() {
    if (unitSpecArr && structSpecArr) {
        //startIterating();
    }
}
/**
     * returns a strongly typed bound function
     * @param {(index: number, err, dat) => any} f
     * @returns {(err, dat) => any}
     */
function strongBind(i, f) {
    return f.bind(null, i);
}
/**
 * returns native type casted value of input string
 * @param {string} str
 * @param {*} unusedArgs
 * @returns {*}
 */
function stringCast(str, ...unusedArgs) {
    let float = parseFloat(str);
    if (!isNaN(float)) {
        return float;
    }
    let bool = str.toLowerCase();
    if (bool === "false") {
        return false;
    }
    else if (bool === "true") {
        return true;
    }
    let obj;
    try {
        obj = JSON.parse(str);
    }
    catch (error) {
        return str;
    }
    return obj;
}
var readCSV = {
    filePaths: [unitSpecFile, structSpecFile],
    callbacks: [onUnitSpecReadComplete, onStructSpecReadComplete]
};
for (let i = readCSV.filePaths.length - 1; i >= 0; i--) {
    let fileName = readCSV.filePaths[i];
    fs.readFile(fileName, strongBind(i, (i, err, data) => {
        if (err) {
            console.error(err.message);
        }
        else {
            csv(data.toString(), {
                cast: stringCast,
                columns: true,
                skip_empty_lines: true,
                skip_lines_with_empty_values: true
            }, strongBind(i, (i, err, data) => {
                if (err) {
                    console.error(err.message);
                }
                else {
                    console.log(JSON.stringify(data));
                    readCSV.callbacks[i](data);
                }
            }));
        }
    }));
}
var [xpos, ypos, time, unitType] = [0, 0, 0, 0];
/*
const structSpec: Array<{ name: string, hp: number, dps: number, rangeMax: number, rangeMin: number, splash:number, tgts: Array<number> }> = [
    {
        name: 'cannon',
        hp: 300,
        dps: 50,
        rangeMax: 10,
        rangeMin:0,
        splash: 0,
        tgts:[0]
    },
    {
        name: 'archtowr',
        hp: 300,
        dps: 50,
        rangeMax: 10,
        rangeMin:0,
        splash: 0,
        tgts: [0,1]
    },
    {
        name: 'mortar',
        hp: 300,
        dps: 50,
        rangeMax: 13,
        rangeMin:3,
        splash: 1,
        tgts: [0, 1]
    },
]; */
var estimateVector = tf.variable(tf.tensor2d([[0, 0, 0, 0], [0, 0, 0, 0]]));
function predict() {
}
tf.tidy(() => {
    tf.tensor([], [0, 0], "int32");
});
//# sourceMappingURL=LinearRegressor.js.map