import { Page, expect } from '@playwright/test';

export class BasePage {
  constructor(public page: Page) {}

  async goto(url: string) {
    await this.page.goto(url);
    await expect(this.page).toHaveTitle(/.*/);
  }

  async verifyTitle() {
    const title = await this.page.title();
    console.log('Title:', title);
    expect(title.length).toBeGreaterThan(0);
  }

  async clickElement(selector: string) {
    await this.page.locator(selector).click();
  }

  async typeInto(selector: string, value: string) {
    await this.page.fill(selector, value);
  }
}
