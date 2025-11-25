# Setting Up Custom Domain with GoDaddy for GitHub Pages (CNAME Method)

## Step 1: Update the CNAME File

Replace the content in `public/CNAME` with your subdomain:

```text
www.yourdomain.com
```

**Example:** If your domain is `example.com`, use `www.example.com`

---

## Step 2: Configure DNS in GoDaddy

1. Log in to your GoDaddy account
2. Go to **My Products** → **DNS** for your domain
3. Add a **CNAME record**:
   - **Type:** CNAME
   - **Name:** www
   - **Value:** nick-rajwade.github.io
   - **TTL:** 600 seconds (or default)

That's it! Just one DNS record needed.

---

## Step 3: Commit and Push

```bash
# First, edit public/CNAME with your actual domain (e.g., www.example.com)
git add public/CNAME vite.config.js .github/workflows/deploy.yml
git commit -m "Add custom domain configuration"
git push origin main
```

---

## Step 4: Configure GitHub Pages Settings

1. Go to: `https://github.com/nick-rajwade/ai-training/settings/pages`
2. Under **Custom domain**, enter your subdomain (e.g., `www.yourdomain.com`)
3. Wait a few minutes for DNS check to complete
4. Once verified, check **Enforce HTTPS** (this may take a few minutes to become available)

---

## Step 5: Wait for DNS Propagation

DNS changes typically propagate in 10-30 minutes, but can take up to 48 hours.

You can check DNS propagation at: `https://www.whatsmydns.net/`
- Enter your subdomain (e.g., www.yourdomain.com)
- Select "CNAME" record type

---

## Verification

1. Visit your custom domain (e.g., www.yourdomain.com)
2. Verify HTTPS is working (padlock icon in browser)
3. Test that all pages and assets load correctly

---

## Current Configuration

✅ Vite base path set to `/` (for custom domain)
✅ Workflow configured to deploy with GitHub Actions
✅ CNAME file created in `/public/CNAME` (defaults to www.yourdomain.com)

**Next:** Replace `www.yourdomain.com` in `public/CNAME` with your actual subdomain and add the CNAME record in GoDaddy!
