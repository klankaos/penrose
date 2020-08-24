// Utils that are unrelated to the engine, but autodiff/opt/etc only

import { sc } from "./Optimizer";
import { Tensor } from "@tensorflow/tfjs";

// Utils for converting types from AD to numeric

const tup2ADtoNumber = (t: [Tensor, Tensor]): [number, number] =>
    [sc(t[0]), sc(t[1])];

const tup4ADtoNumber = (t: [Tensor, Tensor, Tensor, Tensor]): [number, number, number, number] =>
    [sc(t[0]), sc(t[1]), sc(t[2]), sc(t[3])];

const tup2llistADtoNumber = (xss: [Tensor, Tensor][][]): [number, number][][] =>
    xss.map(xs => xs.map(t => [sc(t[0]), sc(t[1])]));

const floatADtoNumber = (v: IFloatV<Tensor>): IFloatV<number> => {
    return {
        tag: "FloatV",
        contents: sc(v.contents)
    };
};

const ptADtoNumber = (v: IPtV<Tensor>): IPtV<number> => {
    return {
        tag: "PtV",
        contents: tup2ADtoNumber(v.contents)
    };
};

const ptListADtoNumber = (v: IPtListV<Tensor>): IPtListV<number> => {
    return {
        tag: "PtListV",
        contents: v.contents.map(tup2ADtoNumber)
    };
};

const listADtoNumber = (v: IListV<Tensor>): IListV<number> => {
    return {
        tag: "ListV",
        contents: v.contents.map(sc)
    };
};

const tupADtoNumber = (v: ITupV<Tensor>): ITupV<number> => {
    return {
        tag: "TupV",
        contents: tup2ADtoNumber(v.contents)
    };
};

const llistADtoNumber = (v: ILListV<Tensor>): ILListV<number> => {
    return {
        tag: "LListV",
        contents: v.contents.map(e => e.map(sc))
    };
};

const hmatrixADtoNumber = (v: IHMatrixV<Tensor>): IHMatrixV<number> => {
    const m = v.contents;
    return {
        tag: "HMatrixV",
        contents: {
            xScale: sc(m.xScale),
            xSkew: sc(m.xSkew),
            yScale: sc(m.yScale),
            ySkew: sc(m.ySkew),
            dx: sc(m.dx),
            dy: sc(m.dy),
        }
    };
};

const polygonADtoNumber = (v: IPolygonV<Tensor>): IPolygonV<number> => {
    const xs0 = tup2llistADtoNumber(v.contents[0]);
    const xs1 = tup2llistADtoNumber(v.contents[1]);
    const xs2 = [tup2ADtoNumber(v.contents[2][0]), tup2ADtoNumber(v.contents[2][1])] as [[number, number], [number, number]];
    const xs3 = v.contents[3].map(tup2ADtoNumber);

    return {
        tag: "PolygonV",
        contents: [xs0, xs1, xs2, xs3]
    };
};

const colorADtoNumber = (v: Color<Tensor>): Color<number> => {
    if (v.tag === "RGBA") {
        const rgb = v.contents;
        return {
            tag: "RGBA",
            contents: [sc(rgb[0]), sc(rgb[1]), sc(rgb[2]), sc(rgb[3])]
        };
    } else if (v.tag === "HSVA") {
        const hsv = v;
        return {
            tag: "HSVA",
            contents: [sc(hsv[0]), sc(hsv[1]), sc(hsv[2]), sc(hsv[3])]
        };
    } else {
        throw Error("unexpected color tag");
    }
};

const colorVADtoNumber = (v: IColorV<Tensor>): IColorV<number> => {
    return {
        tag: "ColorV",
        contents: colorADtoNumber(v.contents)
    };
};

const paletteADtoNumber = (v: IPaletteV<Tensor>): IPaletteV<number> => {
    return {
        tag: "PaletteV",
        contents: v.contents.map(colorADtoNumber)
    };
};

export const valueAutodiffToNumber = (v: Value<Tensor>): Value<number> => {
    // TODO: Is there a way to write these conversion functions with less boilerplate?
    const nonnumericValueTypes = ["IntV", "BoolV", "StrV", "ColorV", "PaletteV", "FileV", "StyleV"];

    if (v.tag === "FloatV") {
        return floatADtoNumber(v);
    } else if (v.tag === "PtV") {
        return ptADtoNumber(v);
    } else if (v.tag === "PtListV") {
        return ptListADtoNumber(v);
    } else if (v.tag === "ListV") {
        return listADtoNumber(v);
    } else if (v.tag === "TupV") {
        return tupADtoNumber(v);
    } else if (v.tag === "LListV") {
        return llistADtoNumber(v);
    } else if (v.tag === "HMatrixV") {
        return hmatrixADtoNumber(v);
    } else if (v.tag === "PolygonV") {
        return polygonADtoNumber(v);
    } else if (v.tag === "ColorV") {
        return colorVADtoNumber(v);
    } else if (v.tag === "PaletteV") {
        return paletteADtoNumber(v);
    } else if (nonnumericValueTypes.includes(v.tag)) {
        return v as Value<number>;
    } else {
        throw Error(`unimplemented conversion from autodiff types for numeric types for value type '${v.tag}'`);
    }
};
