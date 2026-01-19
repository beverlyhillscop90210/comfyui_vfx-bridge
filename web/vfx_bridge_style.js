import { app } from "../../scripts/app.js";

// VFX Bridge Custom Node Styling
// Clean anthracite theme with "peterschings" watermark

const VFX_BRIDGE_NODES = [
    "EXRHotFolderLoader",
    "MatteChannelSplitter", 
    "MetadataDisplay",
    "EXRSaveNode",
    "PreviewMatte",
    "ChannelSelector",
    "EXRToImage",
    "MaskToImage",
    "OutputTransform"
];

// Anthracite - one clean color
const ANTHRACITE = "#2d2d2d";
const ANTHRACITE_DARK = "#232323";
const WATERMARK = "rgba(255,255,255,0.06)";

app.registerExtension({
    name: "vfx.bridge.style",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!VFX_BRIDGE_NODES.includes(nodeData.name)) {
            return;
        }
        
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        
        nodeType.prototype.onDrawForeground = function(ctx) {
            if (origOnDrawForeground) {
                origOnDrawForeground.apply(this, arguments);
            }
            
            // Watermark bottom right
            ctx.save();
            ctx.font = "7px monospace";
            ctx.fillStyle = WATERMARK;
            ctx.textAlign = "right";
            ctx.fillText("peterschings", this.size[0] - 5, this.size[1] - 3);
            ctx.restore();
        };
        
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }
            
            // Clean anthracite - same for all
            this.color = ANTHRACITE;
            this.bgcolor = ANTHRACITE_DARK;
        };
    },
    
    async setup() {
        console.log("[VFX Bridge] Styling loaded");
    }
});
