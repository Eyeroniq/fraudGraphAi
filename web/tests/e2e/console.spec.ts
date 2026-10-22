import AxeBuilder from "@axe-core/playwright";
import { expect, type Page, test } from "@playwright/test";

// Navigate with the nav (client-side), so the in-browser mock backend keeps its state.
const nav = (page: Page, name: string) => page.getByRole("navigation", { name: "Main" }).getByRole("link", { name, exact: true }).click();

async function championLabels(page: Page): Promise<number> {
  const text = await page.getByText(/champion · \d[\d,]* labels/).first().textContent();
  return Number(text?.match(/(\d[\d,]*) labels/)?.[1].replaceAll(",", ""));
}

test("start the stream, a case appears, resolving it adds a label", async ({ page }) => {
  await page.goto("/drift");
  await page.getByRole("button", { name: /Start/ }).click();
  await expect(page.getByText("Replayer started")).toBeVisible();
  await expect(page.getByText("● running")).toBeVisible();
  await page.waitForTimeout(3_000); // a few replayed steps
  await page.getByRole("button", { name: "Stop" }).click();
  await expect(page.getByText("○ stopped")).toBeVisible();
  // Labels stop arriving with the replayer; wait for a refresh after the last one.
  await page.waitForTimeout(6_000);
  const before = await championLabels(page);

  await nav(page, "Queue");
  await page.getByRole("radio", { name: "Newest" }).click();
  const firstCase = page.locator("tbody tr").first().getByRole("link");
  await expect(firstCase).toBeVisible();
  await firstCase.click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("CASE #");
  await expect(page.getByText("P(fraud)").first()).toBeVisible(); // decision loaded
  await expect(page.getByRole("button", { name: /Fraud/ })).toBeEnabled();
  await page.keyboard.press("f");
  await expect(page.getByText(/^Case #\d+ labelled fraud$/)).toBeVisible();

  await nav(page, "Drift");
  await expect.poll(() => championLabels(page)).toBe(before + 1);
});

test("live page streams decisions", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("demo data")).toBeVisible();
  await expect(page.locator("tbody tr").first()).toBeVisible({ timeout: 10_000 });
});

test("cost sliders redraw the threshold cost", async ({ page }) => {
  await page.goto("/lab");
  const yours = page.getByText(/At your threshold:/);
  await expect(yours).toBeVisible();
  const before = await yours.textContent();
  await page.getByLabel(/Review cost/).fill("15000");
  await expect(yours).not.toHaveText(before ?? "");
});

for (const path of ["/", "/cases", "/cases/view?id=1", "/lab", "/drift", "/explain", "/model", "/score"]) {
  test(`no serious accessibility violations on ${path}`, async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" }); // scan the settled page
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await page.waitForTimeout(800);
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(serious.map((v) => `${v.id}: ${v.nodes.length} × ${v.help}`)).toEqual([]);
  });
}
