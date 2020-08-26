// Utils that are unrelated to the engine, but autodiff/opt/etc only

import { sc } from "./Optimizer";
import { Tensor, scalar } from "@tensorflow/tfjs";
import { mapValues } from 'lodash';

// TODO: Is there a way to write these mapping/conversion functions with less boilerplate?

// Generic utils for mapping over values

function mapTup2<T, S>(
    f: (arg: T) => S,
    t: [T, T]
): [S, S] {
    return [f(t[0]), f(t[1])];
};

function mapTup4<T, S>(
    f: (arg: T) => S,
    t: [T, T, T, T]
): [S, S, S, S] {
    return [f(t[0]), f(t[1]), f(t[2]), f(t[3])];
};

function mapTup2LList<T, S>(
    f: (arg: T) => S,
    xss: [T, T][][]
): [S, S][][] {
    return xss.map(xs => xs.map(t => [f(t[0]), f(t[1])]));
};

// Mapping over values

function mapFloat<T, S>(
    f: (arg: T) => S,
    v: IFloatV<T>
): IFloatV<S> {
    return {
        tag: "FloatV",
        contents: f(v.contents)
    };
};

function mapPt<T, S>(
    f: (arg: T) => S,
    v: IPtV<T>
): IPtV<S> {
    return {
        tag: "PtV",
        contents: mapTup2(f, v.contents) as [S, S]
    };
};

function mapPtList<T, S>(
    f: (arg: T) => S,
    v: IPtListV<T>
): IPtListV<S> {
    return {
        tag: "PtListV",
        contents: v.contents.map(t => mapTup2(f, t))
    };
};

function mapList<T, S>(
    f: (arg: T) => S,
    v: IListV<T>
): IListV<S> {
    return {
        tag: "ListV",
        contents: v.contents.map(f)
    };
};

function mapTup<T, S>(
    f: (arg: T) => S,
    v: ITupV<T>
): ITupV<S> {
    return {
        tag: "TupV",
        contents: mapTup2(f, v.contents)
    };
};

function mapLList<T, S>(
    f: (arg: T) => S,
    v: ILListV<T>
): ILListV<S> {
    return {
        tag: "LListV",
        contents: v.contents.map(e => e.map(f))
    };
};

function mapHMatrix<T, S>(
    f: (arg: T) => S,
    v: IHMatrixV<T>
): IHMatrixV<S> {
    const m = v.contents;
    return {
        tag: "HMatrixV",
        contents: { // TODO: This could probably be a generic map over object values
            xScale: f(m.xScale),
            xSkew: f(m.xSkew),
            yScale: f(m.yScale),
            ySkew: f(m.ySkew),
            dx: f(m.dx),
            dy: f(m.dy),
        }
    };
};

function mapPolygon<T, S>(
    f: (arg: T) => S,
    v: IPolygonV<T>
): IPolygonV<S> {
    const xs0 = mapTup2LList(f, v.contents[0]);
    const xs1 = mapTup2LList(f, v.contents[1]);
    const xs2 = [mapTup2(f, v.contents[2][0]), mapTup2(f, v.contents[2][1])] as [[S, S], [S, S]];
    const xs3 = v.contents[3].map(e => mapTup2(f, e));

    return {
        tag: "PolygonV",
        contents: [xs0, xs1, xs2, xs3]
    };
};

function mapColorInner<T, S>(
    f: (arg: T) => S,
    v: Color<T>
): Color<S> {
    if (v.tag === "RGBA") {
        const rgb = v.contents;
        return {
            tag: "RGBA",
            contents: mapTup4(f, rgb)
        };
    } else if (v.tag === "HSVA") {
        const hsv = v.contents;
        return {
            tag: "HSVA",
            contents: mapTup4(f, hsv)
        };
    } else {
        throw Error("unexpected color tag");
    }
};

function mapColor<T, S>(
    f: (arg: T) => S,
    v: IColorV<T>
): IColorV<S> {
    return {
        tag: "ColorV",
        contents: mapColorInner(f, v.contents)
    };
};

function mapPalette<T, S>(
    f: (arg: T) => S,
    v: IPaletteV<T>
): IPaletteV<S> {
    return {
        tag: "PaletteV",
        contents: v.contents.map(e => mapColorInner(f, e))
    };
};

// Utils for converting types of values

// Expects `f` to be a function between numeric types (e.g. number -> Tensor, Tensor -> number, AD var -> Tensor ...)
// Coerces any non-numeric types
export function mapValueNumeric<T, S>(
    f: (arg: T) => S,
    v: Value<T>
): Value<S> {
    const nonnumericValueTypes = ["IntV", "BoolV", "StrV", "ColorV", "PaletteV", "FileV", "StyleV"];

    if (v.tag === "FloatV") {
        return mapFloat(f, v);
    } else if (v.tag === "PtV") {
        return mapPt(f, v);
    } else if (v.tag === "PtListV") {
        return mapPtList(f, v);
    } else if (v.tag === "ListV") {
        return mapList(f, v);
    } else if (v.tag === "TupV") {
        return mapTup(f, v);
    } else if (v.tag === "LListV") {
        return mapLList(f, v);
    } else if (v.tag === "HMatrixV") {
        return mapHMatrix(f, v);
    } else if (v.tag === "PolygonV") {
        return mapPolygon(f, v);
    } else if (v.tag === "ColorV") {
        return mapColor(f, v);
    } else if (v.tag === "PaletteV") {
        return mapPalette(f, v);
    } else if (nonnumericValueTypes.includes(v.tag)) {
        return v as Value<S>;
    } else {
        throw Error(`unimplemented conversion from autodiff types for numeric types for value type '${v.tag}'`);
    }
};

export const valueAutodiffToNumber = (v: Value<Tensor>): Value<number> => mapValueNumeric(sc, v);

export const valueNumberToAutodiff = (v: Value<number>): Value<Tensor> => mapValueNumeric(scalar, v);

// Walk translation to convert all TagExprs (tagged Done or Pending) in the state to Tensors
// (This is because, when decoded from backend, it's not yet in Tensor form -- although this code could be phased out if the translation becomes completely generated in the frontend)
// TODO: Make this generic, to map over translation

export function mapTagExpr<T, S>(
    f: (arg: T) => S,
    e: TagExpr<T>
): TagExpr<S> {
    if (e.tag === "Done") {
        return {
            tag: "Done",
            contents: mapValueNumeric(f, e.contents)
        };
    } else if (e.tag === "Pending") {
        return {
            tag: "Pending",
            contents: mapValueNumeric(f, e.contents)
        };
    } else if (e.tag === "OptEval") {
        // TODO: Should we map over this anyway?
        console.error("not tag expr; did not convert OptEval Expr");
        return e;
    } else {
        throw Error("unrecognized tag");
    }
};

export function mapGPIExpr<T, S>(
    f: (arg: T) => S,
    e: GPIExpr<T>
): GPIExpr<S> {
    const propDict = Object.entries(e[1])
        .map(([prop, val]) =>
            [prop, mapTagExpr(f, val)]
        );

    return [e[0], Object.fromEntries(propDict)];
};

export function mapTranslation<T, S>(
    f: (arg: T) => S,
    trans: Translation
): Translation {
    const newTrMap = Object.entries(trans.trMap)
        .map(([name, fdict]) => {

            const fdict2 = Object.entries(fdict)
                .map(([prop, val]) => {
                    if (val.tag === "FExpr") {
                        return [prop, { tag: "FExpr", contents: mapTagExpr(f, val.contents) }];
                    } else if (val.tag === "FGPI") {
                        return [prop, { tag: "FGPI", contents: mapGPIExpr(f, val.contents) }];
                    } else {
                        throw Error("unknown tag on field expr");
                    }
                });

            return [name, Object.fromEntries(fdict2)];
        });

    return {
        ...trans,
        trMap: Object.fromEntries(newTrMap)
    };
};

// TODO: Did this actually fix the original problem with color conversion?
export const walkTranslationConvert = (trans: Translation): Translation => {
    return mapTranslation(scalar, trans);
};
