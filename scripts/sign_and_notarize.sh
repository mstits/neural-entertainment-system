#!/usr/bin/env bash
# Code-sign, notarize, and staple the NES .app bundle for
# distribution outside the App Store.
#
# Prerequisites (set in your shell or a .env.local you source):
#   APPLE_DEVELOPER_ID        - e.g. "Developer ID Application: Your Name (TEAM1234)"
#   APPLE_ID                  - your Apple ID email
#   APPLE_APP_PASSWORD        - app-specific password from appleid.apple.com
#   APPLE_TEAM_ID             - 10-char team identifier
#
# Usage:
#   ./scripts/build_app.sh          # builds dist/main.app
#   ./scripts/sign_and_notarize.sh  # signs + notarizes + staples + makes DMG
#
# Output: dist/NES.dmg ready to distribute.

set -euo pipefail

APP="dist/main.app"
RENAMED="dist/NES.app"
DMG="dist/NES.dmg"

: "${APPLE_DEVELOPER_ID:?set APPLE_DEVELOPER_ID to your 'Developer ID Application: ... (TEAMID)' identity}"
: "${APPLE_ID:?set APPLE_ID to your Apple ID email}"
: "${APPLE_APP_PASSWORD:?set APPLE_APP_PASSWORD to an app-specific password}"
: "${APPLE_TEAM_ID:?set APPLE_TEAM_ID to your 10-char team ID}"

if [[ ! -d "$APP" ]]; then
    echo "ERROR: $APP not found. Run ./scripts/build_app.sh first." >&2
    exit 1
fi

# Rename to a human-friendly app name
rm -rf "$RENAMED"
cp -R "$APP" "$RENAMED"

echo "==> Code-signing $RENAMED"
# Deep sign every embedded binary (Python framework, dylibs, .so, etc.)
codesign --force --deep --options=runtime \
    --sign "$APPLE_DEVELOPER_ID" \
    --timestamp \
    "$RENAMED"

echo "==> Verifying signature"
codesign --verify --deep --strict --verbose=2 "$RENAMED"

echo "==> Building zip for notarization"
ZIP="dist/NES.zip"
rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "$RENAMED" "$ZIP"

echo "==> Submitting to Apple for notarization (this can take a few minutes)"
xcrun notarytool submit "$ZIP" \
    --apple-id "$APPLE_ID" \
    --password "$APPLE_APP_PASSWORD" \
    --team-id "$APPLE_TEAM_ID" \
    --wait

echo "==> Stapling the notarization ticket to the app"
xcrun stapler staple "$RENAMED"
xcrun stapler validate "$RENAMED"

echo "==> Building DMG"
rm -f "$DMG"
hdiutil create -volname "NES" \
    -srcfolder "$RENAMED" \
    -ov -format UDZO \
    "$DMG"

echo "==> Signing the DMG"
codesign --force --sign "$APPLE_DEVELOPER_ID" --timestamp "$DMG"

echo ""
echo "Done: $DMG"
echo "Distribute this file. Users on macOS 10.15+ will see a first-launch"
echo "prompt then clean-open behavior."
