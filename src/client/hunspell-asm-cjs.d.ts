// The worker imports hunspell-asm's CJS entry by explicit path (its ESM build is
// broken under bundlers — see vite.config.ts). That subpath has no colocated
// declarations, so map it to the package's real types.
declare module "hunspell-asm/dist/cjs/index.js" {
  export * from "hunspell-asm";
}
