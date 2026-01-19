import { app } from "../../scripts/app.js";

// VFX Bridge Custom Node Styling
// Anthracite dark theme with "peterschings" watermark

const VFX_BRIDGE_NODES = [
    "EXRHotFolderLoader",
    "MatteChannelSplitter", 
    "MetadataDisplay",
    "EXRSaveNode",
    "PreviewMatte",
    "ChannelSelector",
    "EXRToImage",
    "MaskToImage"
];

// Color scheme - Anthracite/Dark Gray
const COLORS = {
    header: "#2a2a2a",           // Dark anthracite header
    headerText: "#e0e0e0",       // Light gray text
    body: "#1e1e1e",             // Darker body
    accent: "#4a90a4",           // Subtle teal accent
    watermark: "rgba(255,255,255,0.08)", // Very subtle watermark
    border: "#3a3a3a"            // Subtle border
};

app.registerExtension({
    name: "vfx.bridge.style",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only style VFX Bridge nodes
        if (!VFX_BRIDGE_NODES.includes(nodeData.name)) {
            return;
        }
        
        // Store original onDrawForeground
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        
        nodeType.prototype.onDrawForeground = function(ctx) {
            // Call original if exists
            if (origOnDrawForeground) {
                origOnDrawForeground.apply(this, arguments);
            }
            
            // Draw watermark in bottom right corner
            ctx.save();
            ctx.font = "8px monospace";
            ctx.fillStyle = COLORS.watermark;
            ctx.textAlign = "right";
            ctx.fillText("peterschings", this.size[0] - 6, this.size[1] - 4);
            ctx.restore();
        };
        
        // Custom colors
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }
            
            // Set custom colors
            this.color = COLORS.body;
            this.bgcolor = COLORS.header;
            
            // Add subtle colored top bar based on node type
            if (nodeData.name === "EXRHotFolderLoader") {
                this.color = "#1a2a1a";  // Subtle green tint
            } else if (nodeData.name === "EXRSaveNode") {
                this.color = "#2a1a1a";  // Subtle red tint
            } else if (nodeData.name.includes("Channel") || nodeData.name.includes("Matte")) {
                this.color = "#1a1a2a";  // Subtle blue tint
            } else if (nodeData.name.includes("Image") || nodeData.name.includes("Mask")) {
                this.color = "#2a2a1a";  // Subtle yellow tint
            }
        };
    },
    
    async setup() {
        // Add custom CSS for VFX Bridge category styling
        const style = document.createElement('style');
        style.textContent = `
            /* VFX Bridge category in menu */
            .litemenu-entry:has(> .litemenu-entry-text[data-content*="VFX Bridge"]),
            .litemenu-entry[data-content*="VFX Bridge"] {
                border-left: 3px solid #4a90a4 !important;
            }
        `;
        document.head.appendChild(style);
        
        console.log("[VFX Bridge] Custom styling loaded - by peterschings");
    }
});
