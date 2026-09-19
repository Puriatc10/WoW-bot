--[[
WoWBotBridge
------------
خروجی وضعیت بازیکن/تارگت برای خواندن توسط پایتون - با انتخاب خودکار بهترین روش:

  حالت A (ترجیحی): اگه io library روی این کلاینت باز باشه، هر تیک یه فایل کوچیک
                    (wbb_state.txt کنار همین ادان) رو بازنویسی می‌کنه. هیچ اثری
                    روی چت یا صفحه نداره.

  حالت B (فالبک):   اگه io بسته بود، یه چت‌فریم واقعی می‌سازه که هیچ‌وقت نمایش داده
                    نمی‌شه (Hide شده + خارج از صفحه)، و روش قدیمی Console.log
                    (consoleLog 1) رو استفاده می‌کنه. باز هم هیچی توی چت اصلی دیده نمی‌شه.

فقط یک کار دستی لازمه (فقط اگه حالت B فعال شد): /console consoleLog 1  و  /reload
ادان خودش بهت می‌گه کدوم حالت فعاله (یه پیام یک‌بار موقع لود).
]]

local ADDON_NAME = "WoWBotBridge"
local UPDATE_INTERVAL = 0.2

local frame = CreateFrame("Frame", "WoWBotBridgeFrame")
local elapsed_acc = 0

local MODE = nil          -- "file" یا "chatlog"
local file_handle = nil
local hidden_chat_frame = nil

local function safe(v, default)
	if v == nil then return default end
	return v
end

-- ---------- ساخت خط داده ----------

local function get_target_info()
	if not UnitExists("target") then
		return "tgt:0"
	end

	local name = safe(UnitName("target"), "unknown")
	name = tostring(name):gsub("|", ""):gsub("%s+", "_")

	local hp = 0
	local hpmax = UnitHealthMax("target")
	if hpmax and hpmax > 0 then
		hp = math.floor((UnitHealth("target") / hpmax) * 100)
	end

	local reaction = UnitReaction("player", "target")
	local react_str = "unknown"
	if reaction then
		if reaction <= 3 then react_str = "hostile"
		elseif reaction == 4 then react_str = "neutral"
		else react_str = "friendly" end
	end

	local dist = -1

	return string.format("tgt:1|tname:%s|thp:%d|tdist:%d|treact:%s", name, hp, dist, react_str)
end

local function build_line()
	local hp, hpmax = UnitHealth("player"), UnitHealthMax("player")
	local hp_pct = (hpmax and hpmax > 0) and math.floor((hp / hpmax) * 100) or 0

	local power, powermax = UnitPower("player"), UnitPowerMax("player")
	local power_pct = (powermax and powermax > 0) and math.floor((power / powermax) * 100) or 0

	local mx, my = 0, 0
	if SetMapToCurrentZone then SetMapToCurrentZone() end
	if GetPlayerMapPosition then
		mx, my = GetPlayerMapPosition("player")
		mx, my = safe(mx, 0), safe(my, 0)
	end

	local facing = safe(GetPlayerFacing(), 0)
	local in_combat = UnitAffectingCombat("player") and 1 or 0

	return string.format(
		"WBB|t:%d|hp:%d|power:%d|mx:%.4f|my:%.4f|facing:%.4f|combat:%d|%s",
		time(), hp_pct, power_pct, mx, my, facing, in_combat, get_target_info()
	)
end

-- ---------- تشخیص و راه‌اندازی حالت ----------

local function try_setup_file_mode()
	if not (io and io.open) then return false end

	local ok, handle = pcall(io.open, "Interface\\AddOns\\WoWBotBridge\\wbb_state.txt", "w")
	if not ok or not handle then return false end

	-- تست واقعی نوشتن (بعضی کلاینت‌ها io رو نصفه‌نیمه باز می‌ذارن)
	local write_ok = pcall(function()
		handle:write("WBB_TEST\n")
		handle:flush()
	end)
	if not write_ok then
		pcall(function() handle:close() end)
		return false
	end

	file_handle = handle
	return true
end

local function try_setup_hidden_chatframe_mode()
	-- یه چت‌فریم واقعی می‌سازیم (نه یه Frame ساده) تا هوک consoleLog بشناستش،
	-- ولی کاملاً مخفی و خارج از دید نگهش می‌داریم.
	local ok, cf = pcall(FCF_OpenNewWindow, "WBBHiddenLog")
	if not ok or not cf then
		-- فالبک نهایی: از خود DEFAULT_CHAT_FRAME استفاده کن (این یکی دیگه واقعاً همیشه هست)
		hidden_chat_frame = DEFAULT_CHAT_FRAME
		return true
	end

	cf:ClearAllPoints()
	cf:SetPoint("CENTER", UIParent, "CENTER", 0, 0)
	cf:SetAlpha(0)
	cf:EnableMouse(false)
	cf:SetClampedToScreen(false)
	if cf.Hide then cf:Hide() end
	-- بعضی نسخه‌ها نیاز دارن حتی وقتی Hide شده باز AddMessage کار کنه؛ چون خودمون
	-- کنترلش می‌کنیم و هیچ‌جا Show نمی‌کنیمش، برای کاربر هیچ‌وقت دیده نمی‌شه.

	hidden_chat_frame = cf
	return true
end

local function init_mode()
	if try_setup_file_mode() then
		MODE = "file"
		DEFAULT_CHAT_FRAME:AddMessage("|cff00ff00[WoWBotBridge]|r حالت فایل مستقیم فعال شد (تمیزترین حالت، بدون اثر روی چت).")
	elseif try_setup_hidden_chatframe_mode() then
		MODE = "chatlog"
		DEFAULT_CHAT_FRAME:AddMessage("|cffffff00[WoWBotBridge]|r حالت چت‌فریم مخفی فعال شد. اگه هنوز نکردی: /console consoleLog 1  سپس /reload")
	else
		MODE = "visible_chat_fallback"
		DEFAULT_CHAT_FRAME:AddMessage("|cffff0000[WoWBotBridge]|r هیچ روش تمیزی پیدا نشد؛ موقتاً از چت اصلی استفاده می‌شه (ممکنه اسپم بشه).")
	end
end

-- ---------- تیک اصلی ----------

frame:SetScript("OnUpdate", function(self, elapsed)
	if not MODE then return end

	elapsed_acc = elapsed_acc + elapsed
	if elapsed_acc < UPDATE_INTERVAL then return end
	elapsed_acc = 0

	local line = build_line()

	if MODE == "file" and file_handle then
		local ok = pcall(function()
			file_handle:close()
			file_handle = io.open("Interface\\AddOns\\WoWBotBridge\\wbb_state.txt", "w")
			file_handle:write(line, "\n")
			file_handle:flush()
		end)
		if not ok then
			-- اگه یهو io خراب شد، بی‌سروصدا سوییچ کن به حالت چت مخفی
			try_setup_hidden_chatframe_mode()
			MODE = "chatlog"
		end
	elseif hidden_chat_frame then
		hidden_chat_frame:AddMessage(line)
	end
end)

frame:RegisterEvent("ADDON_LOADED")
frame:RegisterEvent("PLAYER_LEAVING_WORLD")
frame:SetScript("OnEvent", function(self, event, name)
	if event == "ADDON_LOADED" and name == ADDON_NAME then
		init_mode()
	elseif event == "PLAYER_LEAVING_WORLD" then
		if file_handle then
			pcall(function() file_handle:close() end)
			file_handle = nil
		end
	end
end)

-- تست دستی
SLASH_WOWBOTBRIDGE1 = "/wbb"
SlashCmdList["WOWBOTBRIDGE"] = function(msg)
	DEFAULT_CHAT_FRAME:AddMessage("[WoWBotBridge] mode=" .. tostring(MODE) .. "  " .. build_line())
end
