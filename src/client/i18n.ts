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
  modeLabel: string;
  modeWeb: string;
  modeLocal: string;
  modeWebExplainer: string;
  modeLocalExplainer: string;
  displayLabel: string;
  displayTop: string;
  displayAll: string;
  displayShowAll: string;
  displayLocalTooltip: string;
  localIdle: string;
  localCheckingCache: string;
  localTierLabel: string;
  localTierLight: string;
  localTierHeavy: string;
  detailsHide: string;
  detailsShow: string;
  localConsentText: string;
  localConsentButton: string;
  localRedownloadText: string;
  localRedownloadButton: string;
  localUpdateAvailable: string;
  localUpdateButton: string;
  localUpdating: string;
  localMetadata: string;
  localVerifyingRuntime: string;
  localModelInfo: string;
  localDownloading: string;
  localReadingCache: string;
  localSessionWorker: string;
  localSessionFallback: string;
  localSessionMain: string;
  localReady: string;
  localFailed: string;
  localRunning: string;
  localBatch: string;
  localDone: string;
  localMemoryLimit: string;
  localTokensPerSecond: string;
  statsButtonLabel: string;
  statsTitle: string;
  statsInferenceMode: string;
  statsModeWorker: string;
  statsModeMain: string;
  statsModeUnknown: string;
  statsMemory: string;
  statsLastRun: string;
  statsLastRunEmpty: string;
  statsModel: string;
  statsModelVersion: string;
  statsCache: string;
  statsRedownload: string;
  cacheHit: string;
  cacheMiss: string;
  cacheStored: string;
  cacheFailed: string;
  cacheUnavailable: string;
  unknownSize: string;
  copyButton: string;
  copied: string;
  resultHeading: string;
  resultEmpty: string;
  legendLabel: string;
  legendResolved: string;
  legendAmbiguous: string;
  legendUser: string;
  legendUnknown: string;
  primerLink: string;
  primerTitle: string;
  primerIntro: string;
  primerGraveName: string;
  primerGraveDesc: string;
  primerGraveEx: string;
  primerAcuteName: string;
  primerAcuteDesc: string;
  primerAcuteEx: string;
  primerTildeName: string;
  primerTildeDesc: string;
  primerTildeEx: string;
  primerMixed: string;
  primerPair: string;
  primerMore: string;
  taggerNotice: string;
  unknownTitle: string;
  variantsLoading: string;
  variantsNone: string;
  variantsError: string;
  errEmpty: string;
  errTooLong: string;
  errUpstream: string;
  errUnexpected: string;
  footerProject: string;
};

export const UI: Record<Lang, UiStrings> = {
  lt: {
    tagline: "Įklijúokite lietùvišką tèkstą — gáusite pilnaĩ sukirčiúotą.",
    inputLabel: "Tèkstas",
    accentButton: "Sukirčiúoti",
    accentButtonLoading: "Kirčiúojama...",
    modeLabel: "Režìmas",
    modeWeb: "Internetù",
    modeLocal: "Viẽtinis",
    modeWebExplainer:
      "Tekstas siunčiamas į serverį — kirčiuoja VDU kirčiuoklė (kalbu.vdu.lt), morfologiją žymi UDPipe (LINDAT).",
    modeLocalExplainer:
      "Vìskas vỹksta jū́sų naršỹklėje põ vienkartìnio mòdelio atsisiuntìmo (~{size}); tèkstas nepaliẽka į́renginio.",
    displayLabel: "Ródymas",
    displayTop: "geriáusi",
    displayAll: "visì",
    displayShowAll: "ródyti visùs",
    displayLocalTooltip:
      "Mòdelis variántus rikiúoja pagal̃ tikimýbę; ródomi tìk >10% variántai.",
    localIdle: "Viẽtinis mòdelis bùs įkéltas pasiriñkus viẽtinį režìmą.",
    localCheckingCache: "Tìkrinama, ar̃ mòdelis jaũ išsáugotas naršỹklėje...",
    localTierLabel: "Mòdelis",
    localTierLight: "Lengvàs",
    localTierHeavy: "Tikslùs",
    detailsHide: "Sutráukti",
    detailsShow: "Išskleĩsti",
    localConsentText:
      "Viẽtiniam kirčiãvimui svetaĩnė víeną kar̃tą atsisiuñčia mòdelį — apiẽ {size} duomenų̃. Vėlesníems apsilañkymams jìs liẽka išsáugotas naršỹklėje.",
    localConsentButton: "Atsisių̃sti mòdelį ({size})",
    localRedownloadText:
      "Mòdelis nebėrà išsáugotas jū́sų naršỹklėje — atsisių̃skite dár kartą.",
    localRedownloadButton: "Atsisių̃sti dár kartą ({size})",
    localUpdateAvailable: "Yrà naũjas mòdelis ({size}).",
    localUpdateButton: "Atnaũjinti mòdelį ({size})",
    localUpdating: "Atnaũjinama...",
    localMetadata: "Skaĩtomi mòdelio metadúomenys...",
    localVerifyingRuntime: "Tìkrinamas WASM runtime: {file} {done}/{total}.",
    localModelInfo: "Mòdelis {size}; talpyklà: {cache}; WASM gìjos: {threads}.",
    localDownloading: "Atsisiuñčiamas mòdelis {done}/{total}.",
    localReadingCache: "Skaĩtomas mòdelis ìš talpỹklos {done}/{total}.",
    localSessionWorker: "Mòdelis inicijúojamas darbìnėje gijojè...",
    localSessionFallback: "Darbìnė gijà neatsãkė; bañdomas atsargìnis režìmas...",
    localSessionMain: "Mòdelis inicijúojamas pagrindìnėje gijojè...",
    localReady: "Viẽtinis mòdelis pàruoštas: {model} · {size} · talpyklà: {cache}.",
    localFailed: "Viẽtinio mòdelio įkélti nepavỹko: {message}",
    localRunning: "Kirčiúojama viẽtiškai: {sentences} sak. · {batches} pak.",
    localBatch:
      "Viẽtiškai: {done}/{sentences} sak. · pak. {batch}/{batches} · {speed} tok./s.",
    localDone:
      "Baĩgta viẽtiškai: {tokens}/{total} tók. · {speed} tok./s · {seconds} s.",
    localMemoryLimit: "Naršỹklės WASM atmintiẽs ribà pasiektà; pérkraukite pùslapį.",
    localTokensPerSecond: "tok./s",
    statsButtonLabel: "WASM statìstika",
    statsTitle: "WASM statìstika",
    statsInferenceMode: "Výkdymas",
    statsModeWorker: "darbìnė gijà",
    statsModeMain: "pagrindìnė gijà",
    statsModeUnknown: "nežìnoma",
    statsMemory: "Atmintìs",
    statsLastRun: "Paskutìnis výkdymas",
    statsLastRunEmpty: "dár nevýkdyta",
    statsModel: "Mòdelis",
    statsModelVersion: "Mòdelio ver̃sija",
    statsCache: "Talpyklà",
    statsRedownload: "Atsisių̃sti iš naujõ",
    cacheHit: "rãsta",
    cacheMiss: "nèrasta",
    cacheStored: "išsáugota",
    cacheFailed: "nepavỹko",
    cacheUnavailable: "nepasiekiamà",
    unknownSize: "nežìnoma",
    copyButton: "Kopijúoti",
    copied: "Nukopijúota ✓",
    resultHeading: "Rezultãtas",
    resultEmpty: "Rezultãtas atsiràs čià.",
    legendLabel: "Žymė́jimas",
    legendResolved: "Pàrinkta pagal̃ kontèkstą",
    legendAmbiguous: "Kelì variántai",
    legendUser: "Jū́sų pasìrinkta",
    legendUnknown: "Žodýne nèrasta",
    primerLink: "Kàs yrà kir̃čio ženklaĩ?",
    primerTitle: "Kir̃čio ženklaĩ",
    primerIntro:
      "Kirčiúotas skiemuõ tãriamas ryškiáusiai. Ilguõsiuose skiemenysè dár skìriama príegaidė — bal̃so „kryptìs“ skiemeñs vidujè. Žymimì trỹs ženklaĩ:",
    primerGraveName: "Kairìnis",
    primerGraveDesc: "trum̃pas kirčiúotas skiemuõ.",
    primerGraveEx: "bùs, vìsas, kàd",
    primerAcuteName: "Dešinìnis",
    primerAcuteDesc:
      "ìlgas skiemuõ, tvirtaprãdė príegaidė — pabrė́žiama skiemeñs pradžià.",
    primerAcuteEx: "výras, káina, áukštas",
    primerTildeName: "Riestìnis",
    primerTildeDesc:
      "ìlgas skiemuõ, tvirtagãlė príegaidė — pabrė́žiama skiemeñs pabaigà.",
    primerTildeEx: "nãmas, laũkas, geraĩ",
    primerMixed:
      "Mišriuosiuose dvigarsiuose (il, ir, al, an…) riestinis rašomas ant l, m, n, r: šil̃tas, var̃das.",
    primerPair:
      "Priegaidė gali skirti žodžius: áušta (švinta) ir aũšta (vėsta).",
    primerMore: "Plačiaũ — VLKK: tartìs ir̃ kirčiãvimas",
    taggerNotice:
      "Kontèkstinė anãlizė nepasiekiamà — dviprãsmiškiems žõdžiams pàrinktos numatýtosios fòrmos.",
    unknownTitle: "Žodýne nèrasta",
    variantsLoading: "Kráunama...",
    variantsNone: "Variántų nèrasta.",
    variantsError: "Variántų gáuti nepavỹko.",
    errEmpty: "Įvèskite tèkstą.",
    errTooLong: "Tèkstas per̃ ìlgas.",
    errUpstream: "Kirčiãvimo paslaugà laĩkinai nepasiekiamà.",
    errUnexpected: "Nepavỹko sukirčiúoti.",
    footerProject: "Ãtvirojo kòdo projèktas",
  },
  en: {
    tagline: "Paste Lithuanian text — get it fully stress-marked.",
    inputLabel: "Text",
    accentButton: "Add stress marks",
    accentButtonLoading: "Working...",
    modeLabel: "Mode",
    modeWeb: "Web",
    modeLocal: "Local",
    modeWebExplainer:
      "Text is sent to the server — accents by VDU kirčiuoklė (kalbu.vdu.lt), morphology tagging by UDPipe (LINDAT).",
    modeLocalExplainer:
      "Everything runs in your browser after a one-time model download (~{size}); nothing leaves this device.",
    displayLabel: "Display",
    displayTop: "top",
    displayAll: "all",
    displayShowAll: "show all",
    displayLocalTooltip:
      "The model ranks labels by probability; only >10% readings are shown.",
    localIdle: "The local model will load when Local mode is selected.",
    localCheckingCache: "Checking whether the model is already saved in this browser...",
    localTierLabel: "Model",
    localTierLight: "Light",
    localTierHeavy: "Accurate",
    detailsHide: "Hide details",
    detailsShow: "Show details",
    localConsentText:
      "To accentuate locally, the site downloads the model once — about {size} of traffic. It stays saved in your browser for future visits.",
    localConsentButton: "Download model ({size})",
    localRedownloadText:
      "The model is no longer saved in your browser — download it again.",
    localRedownloadButton: "Download again ({size})",
    localUpdateAvailable: "Update available — new model ({size}).",
    localUpdateButton: "Update to new model ({size})",
    localUpdating: "Updating...",
    localMetadata: "Reading model metadata...",
    localVerifyingRuntime: "Verifying WASM runtime: {file} {done}/{total}.",
    localModelInfo: "Model {size}; cache: {cache}; WASM threads: {threads}.",
    localDownloading: "Downloading model {done}/{total}.",
    localReadingCache: "Reading model from cache {done}/{total}.",
    localSessionWorker: "Initializing model in a worker...",
    localSessionFallback: "Worker did not respond; trying fallback mode...",
    localSessionMain: "Initializing model on the main thread...",
    localReady: "Local model ready: {model} · {size} · cache: {cache}.",
    localFailed: "Could not load the local model: {message}",
    localRunning: "Running locally: {sentences} sent. · {batches} batches.",
    localBatch:
      "Local: {done}/{sentences} sent. · batch {batch}/{batches} · {speed} tok/s.",
    localDone: "Done locally: {tokens}/{total} tok. · {speed} tok/s · {seconds} s.",
    localMemoryLimit: "Browser WASM memory limit reached; reload the page.",
    localTokensPerSecond: "tok/s",
    statsButtonLabel: "WASM stats",
    statsTitle: "WASM stats",
    statsInferenceMode: "Inference",
    statsModeWorker: "worker",
    statsModeMain: "main thread",
    statsModeUnknown: "unknown",
    statsMemory: "Memory",
    statsLastRun: "Last run",
    statsLastRunEmpty: "not run yet",
    statsModel: "Model",
    statsModelVersion: "Model version",
    statsCache: "Cache",
    statsRedownload: "Re-download",
    cacheHit: "present",
    cacheMiss: "miss",
    cacheStored: "stored",
    cacheFailed: "failed",
    cacheUnavailable: "unavailable",
    unknownSize: "unknown",
    copyButton: "Copy",
    copied: "Copied ✓",
    resultHeading: "Result",
    resultEmpty: "The result will appear here.",
    legendLabel: "Legend",
    legendResolved: "Chosen by context",
    legendAmbiguous: "Multiple variants",
    legendUser: "Chosen by you",
    legendUnknown: "Not in dictionary",
    primerLink: "What do the stress marks mean?",
    primerTitle: "Lithuanian stress marks",
    primerIntro:
      "The stressed syllable is pronounced most prominently. Long syllables additionally carry a pitch contour (priegaidė). Three marks are used:",
    primerGraveName: "Grave",
    primerGraveDesc: "short stressed syllable.",
    primerGraveEx: "bùs, vìsas, kàd",
    primerAcuteName: "Acute",
    primerAcuteDesc:
      "long syllable, falling contour — the start of the syllable is emphasized.",
    primerAcuteEx: "výras, káina, áukštas",
    primerTildeName: "Circumflex",
    primerTildeDesc:
      "long syllable, rising contour — the end of the syllable is emphasized.",
    primerTildeEx: "nãmas, laũkas, geraĩ",
    primerMixed:
      "In mixed diphthongs (il, ir, al, an…) the circumflex sits on the sonorant: šil̃tas, var̃das.",
    primerPair:
      "The contour can distinguish words: áušta (day breaks) vs aũšta (it cools down).",
    primerMore: "More at VLKK: pronunciation and accentuation (in Lithuanian)",
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
    footerProject: "Open-source project",
  },
  ru: {
    tagline: "Вставьте литовский текст — получите его с расставленными ударениями.",
    inputLabel: "Текст",
    accentButton: "Расставить ударения",
    accentButtonLoading: "Обработка...",
    modeLabel: "Режим",
    modeWeb: "Онлайн",
    modeLocal: "Локально",
    modeWebExplainer:
      "Текст отправляется на сервер — ударения ставит VDU kirčiuoklė (kalbu.vdu.lt), морфологию размечает UDPipe (LINDAT).",
    modeLocalExplainer:
      "Всё выполняется в браузере после однократной загрузки модели (~{size}); текст не покидает устройство.",
    displayLabel: "Показ",
    displayTop: "лучшие",
    displayAll: "все",
    displayShowAll: "показать все",
    displayLocalTooltip:
      "Модель ранжирует варианты по вероятности; показаны только варианты >10%.",
    localIdle: "Локальная модель загрузится при выборе локального режима.",
    localCheckingCache: "Проверяем, сохранена ли модель в этом браузере...",
    localTierLabel: "Модель",
    localTierLight: "Лёгкая",
    localTierHeavy: "Точная",
    detailsHide: "Свернуть",
    detailsShow: "Развернуть",
    localConsentText:
      "Для локальной расстановки ударений сайт один раз скачивает модель — около {size} трафика. Она останется сохранённой в браузере для следующих посещений.",
    localConsentButton: "Скачать модель ({size})",
    localRedownloadText:
      "Модель больше не сохранена в браузере — скачайте её снова.",
    localRedownloadButton: "Скачать снова ({size})",
    localUpdateAvailable: "Доступно обновление — новая модель ({size}).",
    localUpdateButton: "Обновить модель ({size})",
    localUpdating: "Обновление...",
    localMetadata: "Чтение метаданных модели...",
    localVerifyingRuntime: "Проверка WASM runtime: {file} {done}/{total}.",
    localModelInfo: "Модель {size}; кеш: {cache}; потоки WASM: {threads}.",
    localDownloading: "Загрузка модели {done}/{total}.",
    localReadingCache: "Чтение модели из кеша {done}/{total}.",
    localSessionWorker: "Инициализация модели в воркере...",
    localSessionFallback: "Воркер не ответил; пробуем резервный режим...",
    localSessionMain: "Инициализация модели в основном потоке...",
    localReady: "Локальная модель готова: {model} · {size} · кеш: {cache}.",
    localFailed: "Не удалось загрузить локальную модель: {message}",
    localRunning: "Локально: {sentences} предл. · {batches} пак.",
    localBatch:
      "Локально: {done}/{sentences} предл. · пак. {batch}/{batches} · {speed} ток./с.",
    localDone:
      "Готово локально: {tokens}/{total} ток. · {speed} ток./с · {seconds} с.",
    localMemoryLimit: "Достигнут лимит WASM-памяти браузера; перезагрузите страницу.",
    localTokensPerSecond: "ток./с",
    statsButtonLabel: "Статистика WASM",
    statsTitle: "Статистика WASM",
    statsInferenceMode: "Выполнение",
    statsModeWorker: "воркер",
    statsModeMain: "основной поток",
    statsModeUnknown: "неизвестно",
    statsMemory: "Память",
    statsLastRun: "Последний запуск",
    statsLastRunEmpty: "ещё не запускалось",
    statsModel: "Модель",
    statsModelVersion: "Версия модели",
    statsCache: "Кеш",
    statsRedownload: "Скачать заново",
    cacheHit: "найден",
    cacheMiss: "нет",
    cacheStored: "сохранён",
    cacheFailed: "ошибка",
    cacheUnavailable: "недоступен",
    unknownSize: "неизвестно",
    copyButton: "Копировать",
    copied: "Скопировано ✓",
    resultHeading: "Результат",
    resultEmpty: "Результат появится здесь.",
    legendLabel: "Обозначения",
    legendResolved: "Выбрано по контексту",
    legendAmbiguous: "Несколько вариантов",
    legendUser: "Выбрано пользователем",
    legendUnknown: "Нет в словаре",
    primerLink: "Что означают знаки ударения?",
    primerTitle: "Литовские знаки ударения",
    primerIntro:
      "Ударный слог произносится наиболее отчётливо. В долгих слогах различается ещё и интонация слога (priegaidė). Используются три знака:",
    primerGraveName: "Гравис",
    primerGraveDesc: "краткий ударный слог.",
    primerGraveEx: "bùs, vìsas, kàd",
    primerAcuteName: "Акут",
    primerAcuteDesc:
      "долгий слог, нисходящая интонация — выделяется начало слога.",
    primerAcuteEx: "výras, káina, áukštas",
    primerTildeName: "Циркумфлекс",
    primerTildeDesc:
      "долгий слог, восходящая интонация — выделяется конец слога.",
    primerTildeEx: "nãmas, laũkas, geraĩ",
    primerMixed:
      "В смешанных дифтонгах (il, ir, al, an…) циркумфлекс ставится на сонорном: šil̃tas, var̃das.",
    primerPair:
      "Интонация различает слова: áušta (светает) и aũšta (остывает).",
    primerMore: "Подробнее — VLKK: произношение и ударение (на литовском)",
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
    footerProject: "Проект с открытым кодом",
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
export type ParallelMorphologyLines = { morphology: string; gloss: string };

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

export function parallelMorphologyLines(
  reading: string,
  lang: Lang,
): ParallelMorphologyLines {
  const dash = reading.indexOf(" - ");
  const mi = (dash === -1 ? reading : reading.slice(0, dash)).trim();
  const meaning = dash === -1 ? "" : reading.slice(dash + 3).trim();
  const morphology: string[] = [];
  const glosses: string[] = [];

  mi.split(", ").forEach((piece) => {
    walkMorphPiece(piece).forEach((token) => {
      const text = token.text.trim();
      if (!text) {
        return;
      }

      morphology.push(text);
      glosses.push(token.gloss ? token.gloss[lang] : text);
    });
  });

  if (meaning) {
    morphology.push(meaning);
    glosses.push(meaning);
  }

  return {
    morphology: morphology.join(" · "),
    gloss: glosses.join(" · "),
  };
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
