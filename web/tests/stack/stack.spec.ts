import { expect, test } from "@playwright/test";

// Against the running compose stack (CI e2e job): the worker replays the fixture
// through the API, a case appears in the console, and labelling it from the keyboard
// stores a label that the API returns.
const API = process.env.STACK_API ?? "http://localhost:8000";
const ADMIN = process.env.STACK_ADMIN_KEY ?? "";

test("replay drives traffic, a case is resolved in the console, the label is stored", async ({ page, request }) => {
  const headers = { "X-API-Key": ADMIN };
  const started = await request.post(`${API}/v1/sim/start`, {
    headers,
    data: { first_step: 1, last_step: 120, steps_per_tick: 12, scenario: "D0" },
  });
  expect(started.ok()).toBeTruthy();

  await expect
    .poll(async () => (await (await request.get(`${API}/v1/cases?status=open&limit=1`, { headers })).json()).total, {
      timeout: 120_000,
      intervals: [2_000],
    })
    .toBeGreaterThan(0);

  await page.goto("/cases");
  await expect(page.getByText("stream live")).toBeVisible({ timeout: 20_000 });
  const first = page.locator("tbody tr").first().getByRole("link");
  await expect(first).toBeVisible();
  const txn = (await first.textContent())?.split("·")[1]?.trim() ?? "";
  await first.click();
  await expect(page.getByText("P(fraud)").first()).toBeVisible();
  await expect(page.getByRole("button", { name: /Fraud/ })).toBeEnabled();
  await page.keyboard.press("f");
  await expect(page.getByText(/^Case #\d+ labelled fraud/)).toBeVisible();

  const detail = await (await request.get(`${API}/v1/transactions/${txn}`, { headers })).json();
  expect(detail.label?.is_fraud).toBe(true);
  await request.post(`${API}/v1/sim/stop`, { headers });
});
