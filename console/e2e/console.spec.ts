/** The review console against the API serving the committed v2 results. Every screen is checked with
 * axe in light and dark; loading, error and empty states are forced by intercepting the API. */
import AxeBuilder from "@axe-core/playwright";
import { type Page, expect, test } from "@playwright/test";

const SCREENS = [
  { hash: "#/releases", heading: "Does the new model hear the care home?" },
  { hash: "#/listen", heading: "Listen to what went wrong" },
  { hash: "#/try", heading: "Dictate a note" },
];

async function axe(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  return r.violations.map((v) => `${v.id}: ${v.nodes.length} (${v.nodes[0]?.target.join(" ")})`);
}

for (const scheme of ["light", "dark"] as const)
  for (const s of SCREENS)
    test(`${s.hash} renders and passes axe (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme, reducedMotion: "reduce" });
      await page.goto(`/${s.hash}`);
      await expect(page.getByRole("heading", { level: 1, name: s.heading })).toBeVisible();
      await expect(page.locator("output[aria-label^='Loading']")).toHaveCount(0, { timeout: 15_000 });
      expect(await axe(page)).toEqual([]);
    });

test("the release table compares every system on the facility notes", async ({ page }) => {
  await page.goto("/#/releases");
  await expect(page.getByRole("rowheader", { name: /Fine-tuned \+ biasing/ })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Residents heard" })).toBeVisible();
  await page.getByLabel("Audio").selectOption("telephone");
  await expect(page.getByRole("rowheader", { name: /Stock Whisper/ }).first()).toBeVisible();
});

test("a note opens to show what was heard, what was lost and the gate's decision", async ({ page }) => {
  await page.goto("/#/listen");
  const first = page.getByRole("button", { expanded: false }).first();
  await first.click();
  await expect(page.getByText("Terms lost")).toBeVisible();
  await expect(page.locator("audio")).toHaveCount(1);
  await expect(page.getByLabel("Heard transcript")).toBeVisible();
});

test("loading shows a skeleton, a failure offers a retry", async ({ page }) => {
  let fail = true;
  await page.route("**/v2/systems", async (r) => {
    await new Promise((x) => setTimeout(x, 400));
    return fail ? r.fulfill({ status: 500, contentType: "application/json", body: '{"detail":"metrics file unreadable"}' }) : r.continue();
  });
  await page.goto("/#/releases");
  await expect(page.locator("output[aria-label='Loading the benchmark']")).toBeVisible();
  await expect(page.getByRole("alert")).toContainText("metrics file unreadable");
  fail = false;
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByRole("heading", { level: 2, name: "Facility notes" })).toBeVisible();
});

test("an empty selection says how to widen it", async ({ page }) => {
  await page.route("**/v2/utterances?**", (r) => r.fulfill({ status: 200, contentType: "application/json", body: '{"total":0,"items":[]}' }));
  await page.goto("/#/listen");
  await expect(page.getByText("No chart errors in this selection.")).toBeVisible();
});

test("dictating a note shows the transcript and whether it is filed or read back", async ({ page }) => {
  await page.route("**/v2/transcribe", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        transcript: "Frau Wiśniewski erhielt um acht Uhr Edoxaban 20 Milligramm.",
        completed_from_facility_list: ["Wiśniewski", "Edoxaban"],
        risk: 0.71, decision: "confirm", vitals: {}, doses: [{ drug: "Edoxaban", amount: 20, unit: "mg" }],
        review_flags: [], timing: { decode_s: 1.2, audio_s: 3.9, rtf: 0.31 },
      }),
    }),
  );
  await page.goto("/#/try");
  await expect(page.getByText("No note yet.")).toBeVisible();
  await page.getByLabel("Recording").setInputFiles({ name: "note.wav", mimeType: "audio/wav", buffer: Buffer.from("RIFF0000WAVE") });
  await page.getByRole("button", { name: "Transcribe" }).click();
  await expect(page.getByText("Read back to the nurse first")).toBeVisible();
  await expect(page.getByText("Edoxaban 20 mg")).toBeVisible();
});

test("works at phone width without sideways scrolling", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#/listen");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});
