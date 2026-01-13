const { chromium } = require('playwright');

(async () => {
  console.log('Starting browser test...');
  
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  
  // Collect console errors
  const errors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });
  
  page.on('pageerror', err => {
    errors.push(err.message);
  });
  
  try {
    // Navigate to the landing page via HTTP server
    console.log('Navigating to landing page...');
    await page.goto('http://localhost:3000', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait for content to load
    await page.waitForTimeout(2000);
    
    // Check if main elements are present
    const title = await page.title();
    console.log('Page title:', title);
    
    // Check for key sections
    const heroText = await page.textContent('h1');
    console.log('Hero section found:', heroText ? 'Yes' : 'No');
    
    // Check for navigation
    const navExists = await page.$('nav');
    console.log('Navigation found:', navExists ? 'Yes' : 'No');
    
    // Report errors
    if (errors.length > 0) {
      console.log('\nConsole errors found:');
      errors.forEach((err, i) => console.log(`  ${i + 1}. ${err}`));
      process.exit(1);
    } else {
      console.log('\nNo console errors found!');
      console.log('Landing page test passed!');
    }
    
  } catch (error) {
    console.error('Test failed:', error.message);
    process.exit(1);
  } finally {
    await browser.close();
  }
})();
