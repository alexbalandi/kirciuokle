// Localization: UI strings and Lithuanian grammar-term translations.
// The morphology abbreviation glosses were authored and reviewed by hand —
// do not regenerate them mechanically.

export type Lang = "lt" | "en" | "ru";

export const LANGS: Lang[] = ["lt", "en", "ru"];

export type UiStrings = {
  tagline: string;
  inputLabel: string;
  accentButton: string;
  accentButtonLoading: string;
  copyButton: string;
  copied: string;
  resultHeading: string;
  resultEmpty: string;
  legendLabel: string;
  legendResolved: string;
  legendAmbiguous: string;
  legendUser: string;
  legendUnknown: string;
  taggerNotice: string;
  unknownTitle: string;
  variantsLoading: string;
  variantsNone: string;
  variantsError: string;
  errEmpty: string;
  errTooLong: string;
  errUpstream: string;
  errUnexpected: string;
  footerData: string;
  footerInspired: string;
};

export const UI: Record<Lang, UiStrings> = {
  lt: {
    tagline: "Įklijuokite lietuvišką tekstą — gausite pilnai sukirčiuotą.",
    inputLabel: "Tekstas",
    accentButton: "Sukirčiuoti",
    accentButtonLoading: "Kirčiuojama...",
    copyButton: "Kopijuoti",
    copied: "Nukopijuota ✓",
    resultHeading: "Rezultatas",
    resultEmpty: "Rezultatas atsiras čia.",
    legendLabel: "Žymėjimas",
    legendResolved: "Parinkta pagal kontekstą",
    legendAmbiguous: "Keli variantai",
    legendUser: "Jūsų pasirinkta",
    legendUnknown: "Žodyne nerasta",
    taggerNotice:
      "Kontekstinė analizė nepasiekiama — dviprasmiškiems žodžiams parinktos numatytosios formos.",
    unknownTitle: "Žodyne nerasta",
    variantsLoading: "Kraunama...",
    variantsNone: "Variantų nerasta.",
    variantsError: "Variantų gauti nepavyko.",
    errEmpty: "Įveskite tekstą.",
    errTooLong: "Tekstas per ilgas.",
    errUpstream: "Kirčiavimo paslauga laikinai nepasiekiama.",
    errUnexpected: "Nepavyko sukirčiuoti.",
    footerData: "Duomenys",
    footerInspired: "įkvėpta",
  },
  en: {
    tagline: "Paste Lithuanian text — get it fully stress-marked.",
    inputLabel: "Text",
    accentButton: "Add stress marks",
    accentButtonLoading: "Working...",
    copyButton: "Copy",
    copied: "Copied ✓",
    resultHeading: "Result",
    resultEmpty: "The result will appear here.",
    legendLabel: "Legend",
    legendResolved: "Chosen by context",
    legendAmbiguous: "Multiple variants",
    legendUser: "Chosen by you",
    legendUnknown: "Not in dictionary",
    taggerNotice:
      "Contextual analysis is unavailable — default forms were chosen for ambiguous words.",
    unknownTitle: "Not in dictionary",
    variantsLoading: "Loading...",
    variantsNone: "No variants found.",
    variantsError: "Could not fetch variants.",
    errEmpty: "Enter some text.",
    errTooLong: "The text is too long.",
    errUpstream: "The accentuation service is temporarily unavailable.",
    errUnexpected: "Accentuation failed.",
    footerData: "Data",
    footerInspired: "inspired by",
  },
  ru: {
    tagline: "Вставьте литовский текст — получите его с расставленными ударениями.",
    inputLabel: "Текст",
    accentButton: "Расставить ударения",
    accentButtonLoading: "Обработка...",
    copyButton: "Копировать",
    copied: "Скопировано ✓",
    resultHeading: "Результат",
    resultEmpty: "Результат появится здесь.",
    legendLabel: "Обозначения",
    legendResolved: "Выбрано по контексту",
    legendAmbiguous: "Несколько вариантов",
    legendUser: "Выбрано пользователем",
    legendUnknown: "Нет в словаре",
    taggerNotice:
      "Контекстный анализ недоступен — для неоднозначных слов выбраны формы по умолчанию.",
    unknownTitle: "Нет в словаре",
    variantsLoading: "Загрузка...",
    variantsNone: "Варианты не найдены.",
    variantsError: "Не удалось получить варианты.",
    errEmpty: "Введите текст.",
    errTooLong: "Текст слишком длинный.",
    errUpstream: "Сервис расстановки ударений временно недоступен.",
    errUnexpected: "Не удалось расставить ударения.",
    footerData: "Данные",
    footerInspired: "вдохновлено",
  },
};

// Grammar abbreviations as emitted by the VDU kirčiuoklė (kalbu.vdu.lt mi
// labels) and kirtis.info (strp inventory). en/ru give the standard
// linguistic terms; lt gives the unabbreviated Lithuanian term.
type Gloss = { lt: string; en: string; ru: string };

const G = (lt: string, en: string, ru: string): Gloss => ({ lt, en, ru });

export const MORPH_GLOSSES: Record<string, Gloss> = {
  // --- part of speech ---
  "dkt.": G("daiktavardis", "noun", "существительное"),
  "dktv.": G("daiktavardis", "noun", "существительное"),
  "bdv.": G("būdvardis", "adjective", "прилагательное"),
  "bdvr.": G("būdvardis", "adjective", "прилагательное"),
  "vksm.": G("veiksmažodis", "verb", "глагол"),
  "dlv.": G("dalyvis", "participle", "причастие"),
  "psdlv.": G("pusdalyvis", "semi-participle (converb)", "полупричастие"),
  "padlv.": G("padalyvis", "gerund (converb)", "деепричастие"),
  "būdn.": G("būdinys", "būdinys (adverbial form)", "бу́динис (наречная форма)"),
  "prv.": G("prieveiksmis", "adverb", "наречие"),
  "prvks.": G("prieveiksmis", "adverb", "наречие"),
  "įv.": G("įvardis", "pronoun", "местоимение"),
  "įvrd.": G("įvardis", "pronoun", "местоимение"),
  "sktv.": G("skaitvardis", "numeral", "числительное"),
  "prl.": G("prielinksnis", "preposition", "предлог"),
  "prlnks.": G("prielinksnis", "preposition", "предлог"),
  "jng.": G("jungtukas", "conjunction", "союз"),
  "jngt.": G("jungtukas", "conjunction", "союз"),
  "dll.": G("dalelytė", "particle", "частица"),
  "jst.": G("jaustukas", "interjection", "междометие"),
  "jstk.": G("jaustukas", "interjection", "междометие"),
  "išt.": G("ištiktukas", "onomatopoeic interjection", "звукоподражание"),
  "ištk.": G("ištiktukas", "onomatopoeic interjection", "звукоподражание"),

  // --- gender ---
  "vyr. g.": G("vyriškoji giminė", "masculine", "мужской род"),
  "vyr.gim.": G("vyriškoji giminė", "masculine", "мужской род"),
  "mot. g.": G("moteriškoji giminė", "feminine", "женский род"),
  "mot.gim.": G("moteriškoji giminė", "feminine", "женский род"),
  "bev. g.": G("bevardė giminė", "neuter", "средний род"),
  "bevrd.gim.": G("bevardė giminė", "neuter", "средний род"),
  "bendr. g.": G("bendroji giminė", "common gender", "общий род"),
  "bendr.gim.": G("bendroji giminė", "common gender", "общий род"),

  // --- number ---
  "vns.": G("vienaskaita", "singular", "ед. число"),
  "vnsk.": G("vienaskaita", "singular", "ед. число"),
  "dgs.": G("daugiskaita", "plural", "мн. число"),
  "dgsk.": G("daugiskaita", "plural", "мн. число"),
  "dvisk.": G("dviskaita", "dual", "двойственное число"),

  // --- case ---
  "vard.": G("vardininkas", "nominative", "именительный"),
  "V.": G("vardininkas", "nominative", "именительный"),
  "kilm.": G("kilmininkas", "genitive", "родительный"),
  "K.": G("kilmininkas", "genitive", "родительный"),
  "naud.": G("naudininkas", "dative", "дательный"),
  "N.": G("naudininkas", "dative", "дательный"),
  "gal.": G("galininkas", "accusative", "винительный"),
  "G.": G("galininkas", "accusative", "винительный"),
  "įnag.": G("įnagininkas", "instrumental", "творительный"),
  "Įn.": G("įnagininkas", "instrumental", "творительный"),
  "viet.": G("vietininkas", "locative", "местный (локатив)"),
  "Vt.": G("vietininkas", "locative", "местный (локатив)"),
  "šauksm.": G("šauksmininkas", "vocative", "звательный"),
  "Š.": G("šauksmininkas", "vocative", "звательный"),

  // --- tense ---
  "es. l.": G("esamasis laikas", "present tense", "настоящее время"),
  "esam.l.": G("esamasis laikas", "present tense", "настоящее время"),
  "būt. k. l.": G("būtasis kartinis laikas", "simple past", "прош. однократное"),
  "būt.kart.l.": G("būtasis kartinis laikas", "simple past", "прош. однократное"),
  "būt. d. l.": G("būtasis dažninis laikas", "past frequentative", "прош. многократное"),
  "būt.d.l.": G("būtasis dažninis laikas", "past frequentative", "прош. многократное"),
  "būt. l.": G("būtasis laikas", "past tense", "прошедшее время"),
  "būt.l.": G("būtasis laikas", "past tense", "прошедшее время"),
  "būs. l.": G("būsimasis laikas", "future tense", "будущее время"),
  "būs.l.": G("būsimasis laikas", "future tense", "будущее время"),

  // --- person ---
  "1 asm.": G("pirmasis asmuo", "1st person", "1-е лицо"),
  "Iasm.": G("pirmasis asmuo", "1st person", "1-е лицо"),
  "2 asm.": G("antrasis asmuo", "2nd person", "2-е лицо"),
  "IIasm.": G("antrasis asmuo", "2nd person", "2-е лицо"),
  "3 asm.": G("trečiasis asmuo", "3rd person", "3-е лицо"),
  "IIIasm.": G("trečiasis asmuo", "3rd person", "3-е лицо"),

  // --- mood ---
  "ties. n.": G("tiesioginė nuosaka", "indicative", "изъявительное накл."),
  "Ties.": G("tiesioginė nuosaka", "indicative", "изъявительное накл."),
  "tar. n.": G("tariamoji nuosaka", "subjunctive", "сослагательное накл."),
  "Tar.": G("tariamoji nuosaka", "subjunctive", "сослагательное накл."),
  "liep. n.": G("liepiamoji nuosaka", "imperative", "повелительное накл."),
  "Liep.": G("liepiamoji nuosaka", "imperative", "повелительное накл."),

  // --- voice / participle type ---
  "veik. r.": G("veikiamoji rūšis", "active", "действительный залог"),
  "veik.r.": G("veikiamoji rūšis", "active", "действительный залог"),
  "neveik. r.": G("neveikiamoji rūšis", "passive", "страдательный залог"),
  "neveik.r.": G("neveikiamoji rūšis", "passive", "страдательный залог"),
  "reikiamyb.": G("reikiamybės dalyvis", "participle of necessity", "причастие долженствования"),

  // --- degree ---
  "nelygin. l.": G("nelyginamasis laipsnis", "positive degree", "положительная степень"),
  "nelygin.": G("nelyginamasis laipsnis", "positive degree", "положительная степень"),
  "aukšt. l.": G("aukštesnysis laipsnis", "comparative", "сравнительная степень"),
  "aukšč. l.": G("aukščiausiasis laipsnis", "superlative", "превосходная степень"),

  // --- reflexivity ---
  "sngr.": G("sangrąžinis", "reflexive", "возвратный"),
  "nesngr.": G("nesangrąžinis", "non-reflexive", "невозвратный"),

  // --- definiteness ---
  "įvardž.": G("įvardžiuotinė forma", "definite (pronominal) form", "местоимённая форма"),
  "neįvardž.": G("neįvardžiuotinė forma", "indefinite form", "простая (неместоимённая) форма"),

  // --- verb forms ---
  "bendr.": G("bendratis", "infinitive", "инфинитив"),
  "bndr.": G("bendratis", "infinitive", "инфинитив"),

  // --- numeral types ---
  "kiekin.": G("kiekinis", "cardinal", "количественное"),
  "kelintin.": G("kelintinis", "ordinal", "порядковое"),
  "daugin.": G("dauginis", "plural-form numeral", "собирательно-множественное"),
  "kuopin.": G("kuopinis", "collective", "собирательное"),

  // --- other ---
  "T.": G("tikrinis daiktavardis", "proper noun", "имя собственное"),
  "sutrmp.": G("sutrumpinimas", "abbreviation", "сокращение"),
};

// Longest-first key list for greedy matching ("būt. k. l." before "būt. l.").
const MORPH_KEYS = Object.keys(MORPH_GLOSSES).sort((a, b) => b.length - a.length);

export type MorphSegment = { text: string; lt?: string };

type MorphToken = { text: string; gloss?: Gloss };

function walkMorphPiece(piece: string): MorphToken[] {
  let rest = piece.trim();
  const tokens: MorphToken[] = [];

  while (rest.length > 0) {
    const key = MORPH_KEYS.find(
      (k) =>
        rest.startsWith(k) &&
        (rest.length === k.length || rest[k.length] === " "),
    );

    if (key) {
      tokens.push({ text: key, gloss: MORPH_GLOSSES[key]! });
      rest = rest.slice(key.length).trimStart();
    } else {
      const space = rest.indexOf(" ");
      if (space === -1) {
        tokens.push({ text: rest });
        rest = "";
      } else {
        tokens.push({ text: rest.slice(0, space) });
        rest = rest.slice(space + 1);
      }
    }
  }

  return tokens;
}

function segmentForToken(token: MorphToken, lang: Lang): MorphSegment {
  if (!token.gloss) {
    return { text: token.text };
  }

  const text = token.gloss[lang];
  return lang === "lt" ? { text } : { text, lt: token.gloss.lt };
}

function readingSegments(reading: string, lang: Lang): MorphSegment[] {
  const pieces = reading.split(", ");
  const segments: MorphSegment[] = [];

  pieces.forEach((piece, pieceIndex) => {
    if (pieceIndex > 0) {
      segments.push({ text: ", " });
    }

    walkMorphPiece(piece).forEach((token, tokenIndex) => {
      if (tokenIndex > 0) {
        segments.push({ text: " " });
      }
      segments.push(segmentForToken(token, lang));
    });
  });

  return segments;
}

/** Translate one mi reading like "bdv., vyr. g., vns. vard." token-wise.
 *  Unrecognized fragments are kept verbatim. */
function translateReading(reading: string, lang: Lang): string {
  return readingSegments(reading, lang)
    .map((segment) => segment.text)
    .join("");
}

/** Translate a full variant info string. Readings are separated by "; ",
 *  and a reading may carry a raw dictionary meaning after " - " — meanings
 *  stay untranslated. */
export function translateMorphology(info: string, lang: Lang): string {
  if (!info) {
    return info;
  }

  return morphologySegments(info, lang)
    .map((segment) => segment.text)
    .join("");
}

export function morphologySegments(info: string, lang: Lang): MorphSegment[] {
  if (!info) {
    return [];
  }

  const segments: MorphSegment[] = [];

  info.split("; ").forEach((segment, segmentIndex) => {
    if (segmentIndex > 0) {
      segments.push({ text: "; " });
    }

    const dash = segment.indexOf(" - ");
    const mi = dash === -1 ? segment : segment.slice(0, dash);
    const meaning = dash === -1 ? "" : segment.slice(dash);
    segments.push(...readingSegments(mi, lang));

    if (meaning) {
      segments.push({ text: meaning });
    }
  });

  return segments;
}

export function detectLang(): Lang {
  const stored = localStorage.getItem("lang");
  if (stored === "lt" || stored === "en" || stored === "ru") {
    return stored;
  }

  const nav = navigator.language?.toLowerCase() ?? "";
  if (nav.startsWith("lt")) {
    return "lt";
  }
  if (nav.startsWith("ru")) {
    return "ru";
  }
  return "en";
}
