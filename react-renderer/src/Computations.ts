import { range } from "lodash";
import { randFloat } from "./Util";
import { Tensor, scalar, stack, cos, sin } from "@tensorflow/tfjs";
import { dist } from "./Constraints"

/**
 * Static dictionary of computation functions
 * TODO: consider using `Dictionary` type so all runtime lookups are type-safe, like here https://codeburst.io/five-tips-i-wish-i-knew-when-i-started-with-typescript-c9e8609029db
 * TODO: think about user extension of computation dict and evaluation of functions in there
 */

// NOTE: These all need to be written in terms of autodiff types (tensors)
export const compDict = {
    rgba: (r: Tensor, g: Tensor, b: Tensor, a: Tensor): IColorV<Tensor> => {
        return {
            tag: "ColorV",
            contents: {
                tag: "RGBA",
                contents: [r, g, b, a],
            },
        };
    },

    hsva: (h: Tensor, s: Tensor, v: Tensor, a: Tensor): IColorV<Tensor> => {
        return {
            tag: "ColorV",
            contents: {
                tag: "HSVA",
                contents: [h, s, v, a],
            },
        };
    },

    // Accepts degrees; converts to radians
    cos: (d: Tensor): Value<Tensor> => {
        console.log("d", d);
        return { tag: "FloatV", contents: cos(d.mul(scalar(Math.PI)).div(scalar(180))) };
    },

    // Accepts degrees; converts to radians
    sin: (d: any) => {
        return { tag: "FloatV", contents: sin(d.mul(scalar(Math.PI)).div(scalar(180))) };
    },

    dot: (v: any, w: any) => {
        const [tv, tw] = [stack(v), stack(w)];
        return { tag: "FloatV", contents: tv.dot(tw) };
    },

    lineLength: ([type, props]: [string, any]) => {
        const [p1, p2] = arrowPts(props);
        return { tag: "FloatV", contents: dist(p1, p2) };
    },

    len: ([type, props]: [string, any]) => {
        const [p1, p2] = arrowPts(props);
        return { tag: "FloatV", contents: dist(p1, p2) };
    },

    // TODO: The functions below need to be written in terms of autodiff types

    orientedSquare: (arr1: any, arr2: any, pt: any, len: number) => {
        throw Error("need to rewrite this function in terms of autodiff types");

        console.log("orientedSquare", arr1, arr2, pt, len);
        const elems = [{ tag: "Pt", contents: [100, 100] },
        { tag: "Pt", contents: [200, 200] },
        { tag: "Pt", contents: [300, 150] }];
        const path = { tag: "Open", contents: elems };
        return { tag: "PathDataV", contents: [path] };
    },

    sampleColor: (alpha: number, colorType: string) => {
        throw Error("need to rewrite this function in terms of autodiff types");

        if (colorType === "rgb") {
            const rgb = range(3).map((_) => randFloat(0.1, 0.9));
            return {
                tag: "ColorV",
                contents: {
                    tag: "RGBA",
                    contents: [...rgb, alpha],
                },
            };
        } else if (colorType === "hsv") {
            const h = randFloat(0, 360);
            return {
                tag: "ColorV",
                contents: {
                    tag: "HSVA",
                    contents: [h, 100, 80, alpha], // HACK: for the color to look good
                },
            };
        } else throw new Error("unknown color type");
    },

};

const arrowPts = ({ startX, startY, endX, endY }: any): [Tensor, Tensor] =>
    [stack([startX.contents, startY.contents]), stack([endX.contents, endY.contents])]

export const checkComp = (fn: string, args: ArgVal<number>[]) => {
    if (!compDict[fn]) throw new Error(`Computation function "${fn}" not found`);
};
