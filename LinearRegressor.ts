import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as csv from 'csv-parse';
import '@tensorflow/tfjs-node';
import * as t from 'runtypes';

const unitSpecFile = './unitSpecs.csv';
const structSpecFile = './structSpecs.csv';
const maxPopulation = 150; //TODO: move to csv?
const unitDeployTimeSlots = 3 * 60 * 5;
//3 minutes of battle time with resolution of 0.2s (approx)

/**
 * since this data is read from an external source, runtime
 * data checking is implemented using the 'runtypes' library
 */

type UnitSpec = t.Static<typeof UnitSpecDef>;
const UnitSpecDef = t.Record({
	id: t.Number,
	name: t.String,
	pop: t.Number,
	hp: t.Number,
	spd: t.Number,
	dps: t.Number,
	range: t.Number,
	isFlying: t.Boolean,
	preferTarget: t.Array(t.Number)
});
var unitSpecArr: Array<UnitSpec>;
var unitSpecCols: UnitSpec | any;//UnitSpec details as obj of arrays
function onUnitSpecReadComplete(data: any) {
	unitSpecArr = t.Array(UnitSpecDef).check(data);
	onReadComplete();
};

type StructSpec = t.Static<typeof StructSpecDef>;
const StructSpecDef = t.Record({
	id: t.Number,
	name: t.String,
	hp: t.Number,
	dps: t.Number,
	rangeMax: t.Number,
	rangeMin: t.Number,
	splash: t.Number,
	targets: t.Array(t.Number),
	size: t.Number,
	maxNum: t.Number,
});
var structSpecArr: Array<StructSpec>;
var structSpecCols: StructSpec | any;//StructSpec details as obj of arrays
function onStructSpecReadComplete(data: any) {
	structSpecArr = t.Array(StructSpecDef).check(data);
	onReadComplete();
};

function onReadComplete() {
	if (unitSpecArr && structSpecArr) {
		createColumnsObject([unitSpecArr, structSpecArr], [unitSpecCols, structSpecCols]);
		//startIterating();
	}
}

/**
 * Convert (a number of) array of dictionaries into dictionaries of arrays
 * @param {any[][]} sourceArrays
 * @param {any[]} targetObjsArray
 */
function createColumnsObject(sourceArray: any[][], targetObjsArray: any[]) {
	for (let i = sourceArray.length - 1; i >= 0; i--) {
		let arrayOfOneType = sourceArray[i];

		let firstDataObj = arrayOfOneType[0];
		let targetObj = targetObjsArray[i];
		for (const colName of firstDataObj) {
			//initialize column names in colsObj with 1st data obj
			targetObj[colName] = [firstDataObj[colName]];
		}
		for (let j = arrayOfOneType.length - 1; j >= 1; j--)//skip 1st obj
		{
			let dataObj = arrayOfOneType[j];
			for (const colName of dataObj) {
				//initialize column names in colsObj with 1st data obj
				targetObj[colName].push(firstDataObj[colName]);
			}
		}
	}
}

/**
	 * returns a strongly typed bound function
	 * @param {(index: number, err, dat) => any} f
	 * @returns {(err, dat) => any}
	 */
function strongBind(i: number, f: (index: number, err, dat) => any): (err, dat) => any {
	return f.bind(null, i);
}

/**
 * returns native type casted value of input string
 * @param {string} str
 * @param {...*} unusedArgs
 * @returns {*}
 */
function stringCast(str: string, ...unusedArgs: any[]): any {
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
	let firstChar = str[0];
	if (firstChar == '{' || firstChar == '[') {
		let obj;
		try {
			obj = JSON.parse(str);
			return obj;
		} catch (error) {
			console.warn('unknown data:' + str);
		}
	}
	return str;

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
				skip_lines_with_empty_values: true,

			}, strongBind(i, (i, err, data) => {
				if (err) {
					console.error(err.message);
				}
				else {
					readCSV.callbacks[i](data);
				}
			}));
		}
	}));
}
var maxMaxStructs = Math.max.apply(Math, structSpecCols.maxNum);

///[unitType,xpos,ypos] * timeSlots
var attackVec = tf.variable(tf.zeros([3, unitDeployTimeSlots], "int32"));

/// [xpos,ypos] * structTypes * structsMax;
var baseLayoutVec2 = tf.variable(tf.zeros([3, structSpecArr.length, maxMaxStructs], "int32"));

///cache constants as tensors for speed
const constsVec: { [k: string]: any } = {};
constsVec.unitPop = tf.tensor1d(unitSpecCols.pop, "int32");
constsVec.ones = tf.ones([2, structSpecArr.length, maxMaxStructs], "bool");
constsVec.structSizes = tf.tensor2d(
	[structSpecCols.size, structSpecCols.size], null, "int32").transpose();
constsVec.structsMask = tf.tensor2d(Array.from({ length: structSpecArr.length },
	(v, k) => {
		let x = [];
		for (let i = maxMaxStructs-1; i >= 0; i--)
		{
			x[i] = (i < structSpecCols[k].maxNum);
		}
		return x;
	}), null, "bool");

/**
 * @returns {number} 0 to 150 (total destruction percentage, +50 for TH)
 */
function predictAttackDamage(): number {
	return tf.tidy(():number => {
		//TODO: benchmark whether squeezing then multiplying or
		//just multiply and get is faster
		let estimatetotalPop:number = attackVec.slice([0, 0], [1, unitDeployTimeSlots])
			.squeeze().mul(constsVec.unitPop).sum().get(0);
		if (estimatetotalPop > maxPopulation) {
			/**
			 * better not to filter out excess units and give explicit loss
			 * instead to prevent "pollution" with large numbers of units
			 */
			return 0;
		}
		/**
		 * filter out units that cannot be placed
		 * Population limit should enforce better placement
		 */
		let [structTypes, structLocs] = tf.split(baseLayoutVec, [1, 2], 0);
		structTypes = structTypes.squeeze();
		structLocs.sub(constsVec.ones)
		throw new Error('Loss simulation failed');
	});
}