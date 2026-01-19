import { app } from "../../scripts/app.js";

const VFX_BRIDGE_NODES = [
    "EXRHotFolderLoader",
    "MatteChannelSplitter", 
    "MetadataDisplay",
    "EXRSaveNode",
    "PreviewMatte",
    "ChannelSelector",
    "EXRToImage",
    "MaskToImage",
    "ColorTransform",
    "DisplayTransform",
    "AOVContactSheet",
    "ShowText"
];

const ANTHRACITE = "#2d2d2d";
const ANTHRACITE_DARK = "#232323";
const WATERMARK = "rgba(255,255,255,0.06)";

app.registerExtension({
    name: "vfx.bridge.style",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!VFX_BRIDGE_NODES.includes(nodeData.name)) return;
        
        if (nodeData.name === "ShowText") {
            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                if (message && message.text) {
                    this.showTextValue = message.text[0] || "";
                    this.setDirtyCanvas(true, true);
                }
            };
            
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (this.showTextValue) {
                    ctx.save();
                    ctx.font = "11px monospace";
                    ctx.fillStyle = "#aaa";
                    ctx.textAlign = "left";
                    const lines = this.showTextValue.split("\n");
                    const lineHeight = 14;
                    const startY = 40;
                    const maxLines = Math.floor((this.size[1] - 50) / lineHeight);
                    for (let i = 0; i < Math.min(lines.length, maxLines); i++) {
                        ctx.fillText(lines[i].substring(0, 40), 10, startY + i * lineHeight);
                    }
                    ctx.restore();
                }
                ctx.save();
                ctx.font = "7px monospace";
                ctx.fillStyle = WATERMARK;
                ctx.textAlign = "right";
                ctx.fillText("peterschings", this.size[0] - 5, this.size[1] - 3);
                ctx.restore();
            };
            
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (origCreated) origCreated.apply(this, arguments);
                this.color = ANTHRACITE;
                this.bgcolor = ANTHRACITE_DARK;
                this.size = [280, 200];
            };
        } else {
            const origDraw = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (origDraw) origDraw.apply(this, arguments);
                ctx.save();
                ctx.font = "7px monospace";
                ctx.fillStyle = WATERMARK;
                ctx.textAlign = "right";
                ctx.fillText("peterschings", this.size[0] - 5, this.size[1] - 3);
                ctx.restore();
            };
            
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (origCreated) origCreated.apply(this, arguments);
                this.color = ANTHRACITE;
                this.bgcolor = ANTHRACITE_DARK;
            };
        }
    },
    
    async setup() {
        console.log("[VFX Bridge] Styling loaded");
    }
});
