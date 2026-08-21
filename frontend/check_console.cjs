const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: "new", args: ['--no-sandbox'] });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  page.on('requestfailed', request => console.log('REQUEST FAILED:', request.url(), request.failure().errorText));

  try {
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle2', timeout: 10000 });
    console.log('Page loaded');
  } catch (err) {
    console.log('Navigation error:', err.message);
  }

  await browser.close();
})();
