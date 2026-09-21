// @ts-check
import { test, expect } from '@playwright/test';

test.describe('PixRevive Web Application E2E Tests', () => {

  test.beforeEach(async ({ page }) => {
    // Ensure standard desktop viewport size for consistent rendering
    await page.setViewportSize({ width: 1280, height: 800 });
  });

  test('Homepage loads correctly with brand and hero elements', async ({ page }) => {
    await page.goto('/');

    // Check title contains PixRevive or frontend
    await expect(page).toHaveTitle(/PixRevive|frontend/i);

    // Brand logo & title in Navbar
    const brandLink = page.getByRole('link', { name: /PixRevive/i }).first();
    await expect(brandLink).toBeVisible();

    // Main Hero headline
    const headline = page.getByRole('heading', { level: 1 });
    await expect(headline).toContainText(/Revive Archival & Damaged Photographs/i);

    // Trust badges
    await expect(page.getByText('100% Local & Offline')).toBeVisible();
    await expect(page.getByText('Zero Cloud Data Uploads')).toBeVisible();
    await expect(page.getByText('Hardware Accelerated (CUDA/CPU)')).toBeVisible();

    // Highlights feature cards
    await expect(page.getByText('4× Super-Resolution')).toBeVisible();
    await expect(page.getByText('Damage Inpainting')).toBeVisible();
    await expect(page.getByText('Deep Colorization')).toBeVisible();
    await expect(page.getByText('Face Enhancement')).toBeVisible();
  });

  test('Navbar displays backend connection status', async ({ page }) => {
    await page.goto('/');

    // Check that backend status pill appears
    const statusPill = page.locator('.status-pill');
    await expect(statusPill).toBeVisible();
    
    // Status text should be either "Backend Online" or "Backend Loading"
    await expect(statusPill).toContainText(/Backend (Online|Loading|Offline)/);
  });

  test('Image uploader and options controls are interactive', async ({ page }) => {
    await page.goto('/');

    // Initial state: Dropzone is visible
    const fileInput = page.locator('input[type="file"]#fileUpload');
    await expect(fileInput).toBeAttached();
    await expect(page.getByText('Select or Drag & Drop an Image')).toBeVisible();
    await expect(page.getByText('Browse Photos')).toBeVisible();

    // Upload a small 1x1 PNG to activate the options workspace
    const samplePngBuffer = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
      'base64'
    );
    await fileInput.setInputFiles({
      name: 'test_sample.png',
      mimeType: 'image/png',
      buffer: samplePngBuffer,
    });

    // Check workspace preview mode is activated
    await expect(page.getByText('1. Original Source')).toBeVisible();
    await expect(page.getByText('2. Restored Output')).toBeVisible();

    // Open settings panel
    const customizeBtn = page.getByRole('button', { name: /Customize AI Pipeline & Device/i });
    await expect(customizeBtn).toBeVisible();
    await customizeBtn.click();

    // Check enhancement option toggles are rendered and interactive
    await expect(page.getByText(/AI Restoration & 4× Upscaling/i)).toBeVisible();
    await expect(page.getByText(/DDColor Chromatic Colorization/i)).toBeVisible();

    // Toggle a checkbox
    const restoreCheckbox = page.locator('#opt-restore');
    await expect(restoreCheckbox).toBeChecked();
    await restoreCheckbox.click();
    await expect(restoreCheckbox).not.toBeChecked();
    await restoreCheckbox.click();
    await expect(restoreCheckbox).toBeChecked();

    // Change Image button returns back to dropzone
    const resetBtn = page.getByRole('button', { name: /Change Image/i });
    await expect(resetBtn).toBeVisible();
    await resetBtn.click();
    await expect(page.getByText('Select or Drag & Drop an Image')).toBeVisible();
  });

  test('Navigation to Documentation page works seamlessly', async ({ page }) => {
    await page.goto('/');

    // Click Documentation link in navbar
    const docLink = page.getByRole('link', { name: /Documentation/i }).first();
    await expect(docLink).toBeVisible();
    await docLink.click();

    // Verify URL & Header
    await expect(page).toHaveURL(/\/documentation/);
    await expect(page.getByRole('heading', { level: 1 })).toContainText(/PixRevive Engine Documentation/i);
    await expect(page.getByText(/Restoration Pipeline Overview/i)).toBeVisible();

    // Click Back to Workspace
    const backLink = page.getByRole('link', { name: /Back to Workspace/i });
    await expect(backLink).toBeVisible();
    await backLink.click();

    await expect(page).toHaveURL('/');
  });

  test('Navigation to Weights & Dataset Admin panel works', async ({ page }) => {
    await page.goto('/');

    // Click Admin link in navbar
    const adminLink = page.getByRole('link', { name: /Admin/i }).first();
    await expect(adminLink).toBeVisible();
    await adminLink.click();

    // Verify URL & Admin page elements
    await expect(page).toHaveURL(/\/developer\/weights/);
    await expect(page.getByRole('heading', { level: 1 })).toContainText(/Weights & Dataset Administration/i);

    // Check return button "Back to Application"
    const backBtn = page.getByRole('button', { name: /Back to Application/i });
    await expect(backBtn).toBeVisible();
    await backBtn.click();
    await expect(page).toHaveURL('/');
  });

  test('Triple-click on Camera Logo secret navigates to Admin panel', async ({ page }) => {
    await page.goto('/');

    const cameraButton = page.locator('button[aria-label="PixRevive Home Logo"]');
    await expect(cameraButton).toBeVisible();

    // Triple click rapidly
    await cameraButton.click({ clickCount: 3 });

    // Verify redirection to /developer/weights
    await expect(page).toHaveURL(/\/developer\/weights/);
  });

  test('Footer renders required brand links and details', async ({ page }) => {
    await page.goto('/');

    // Footer copyright & tagline
    await expect(page.getByText(/Offline Neural Image Restoration & Colorization/i)).toBeVisible();
    await expect(page.getByText(/PixRevive AI/i)).toBeVisible();
  });
});
